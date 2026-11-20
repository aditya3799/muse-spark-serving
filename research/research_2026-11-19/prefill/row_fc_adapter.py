"""Explicitly registered draft-FC row adapter; does not modify or auto-enable SGLang.

Only call this adapter from ordinary replicated-input TP prefill. The model's
original full-M hidden_norm, distinct rank-owned KV weights, positions and cache
write path stay outside it. Strict mode shadows native bits; empirical mode is
a separately declared, runtime-qualified experiment, not a reproducibility proof.
Destroy captured graphs before close(). Never register inside capture.
"""
import ctypes as C
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent

def _library():
    lib=C.CDLL(str(HERE/'row_plans.so'));p=C.c_void_p;i=C.c_int;u=C.c_uint64
    for name,args,result in [('make_row_plan',[i]*5+[u,i],p),('row_plan_count',[p],i),
        ('row_plan_attr',[p,i,i],i),('row_plan_call',[p,i,i,p,p,p,p,u,p],i),('row_plan_destroy',[p],None)]:
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
    return lib

def _workspace():
    backend=str(torch.backends.cuda.preferred_blas_library()).split('.')[-1].lower()
    symbol={'cublas':'_ZN2at4cuda22getChosenWorkspaceSizeEv','cublaslt':'_ZN2at4cuda26getCUDABlasLtWorkspaceSizeEv'}.get(backend)
    if symbol is None:return None
    lib=C.CDLL(str(Path(torch.__file__).resolve().parent/'lib/libtorch_cuda.so'))
    call=getattr(lib,symbol);call.argtypes=[];call.restype=C.c_size_t
    return backend,int(call())

def _policy():
    return (str(torch.backends.cuda.preferred_blas_library()),
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())

def _identity(w):
    try:version=w._version
    except RuntimeError:version=None  # inference tensors; re-register on weight reload
    return w.data_ptr(),version,tuple(w.shape),tuple(w.stride()),w.dtype,w.device
def _stream(x):return torch.cuda.current_stream(x.device).cuda_stream if x.is_cuda else None

def eligible(x,w,world):
    return (world in (2,4,8) and x.is_cuda and w.is_cuda and x.device==w.device
            and x.dtype==w.dtype==torch.bfloat16 and x.ndim==w.ndim==2
            and x.shape[0]>=512 and x.shape[0]%world==0 and x.shape[1]==w.shape[1]
            and x.is_contiguous() and w.is_contiguous() and min(w.shape)>=1024
            and w.shape[0]%128==0 and w.shape[1]%128==0
            and x.data_ptr()%256==0 and w.data_ptr()%256==0)

class RowPlan:
    def __init__(self,lib,handle,world,x,w,workspace_bytes):
        self.lib,self.handle,self.world=lib,handle,world
        self.shape=tuple(x.shape);self.n=w.shape[0];self.device=x.device
        self.identity=_identity(w);self.stream=_stream(x);self.policy=_policy()
        self.scratch=torch.empty(workspace_bytes,dtype=torch.uint8,device=x.device)
        self.index=None
    def invoke(self,index,full,x,w,y):
        return self.lib.row_plan_call(self.handle,index,int(full),w.data_ptr(),x.data_ptr(),y.data_ptr(),self.scratch.data_ptr(),self.scratch.numel(),_stream(x))
    def local(self,x,w,rank):
        if not eligible(x,w,self.world) or tuple(x.shape)!=self.shape or _identity(w)!=self.identity or _stream(x)!=self.stream or _policy()!=self.policy:return None
        count=x.shape[0]//self.world
        y=torch.empty((count,self.n),device=x.device,dtype=x.dtype)
        status=self.invoke(self.index,False,x.narrow(0,rank*count,count),w,y)
        return y if status==0 else None
    def close(self):
        if self.handle:self.lib.row_plan_destroy(self.handle);self.handle=None

@torch.no_grad()
def discover(examples,world):
    if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is not True:
        return None,{'reason':'native BF16 reduction policy requires an unsupported Lt preference mask'}
    examples=list(examples)
    if not examples:return None,{'reason':'no examples'}
    x,w=examples[0]
    if not all(eligible(v,wt,world) and v.shape==x.shape and wt.shape==w.shape for v,wt in examples):return None,{'reason':'unsupported geometry'}
    if torch.cuda.is_current_stream_capturing():return None,{'reason':'registration inside capture'}
    try:lib=_library();backend,bytes_=_workspace()
    except (OSError,AttributeError,TypeError):return None,{'reason':'native ABI unavailable'}
    m,k=x.shape;n=w.shape[0]
    handle=lib.make_row_plan(m,m//world,n,k,1,bytes_,16)
    if not handle:return None,{'reason':'no legal parent/local plan'}
    plan=RowPlan(lib,handle,world,x,w,bytes_)
    props=torch.cuda.get_device_properties(x.device)
    metadata=dict(backend=backend,workspace_bytes=bytes_,shape=[m,n,k],world=world,tried=[],
                  torch=str(torch.__version__),cuda_runtime=torch.version.cuda,device=props.name,
                  capability=[props.major,props.minor],sm_count=props.multi_processor_count,policy=_policy())
    try:
        local=torch.empty((m//world,n),device=x.device,dtype=x.dtype)
        for index in range(lib.row_plan_count(handle)):
            attrs=[lib.row_plan_attr(handle,index,f) for f in range(9)];good=True
            for value,weight in examples:
                reference=F.linear(value,weight)
                # Match every used local result directly against the original
                # full operation. Avoid retaining an additional full-M Lt
                # output solely for diagnostics during production startup.
                for start in range(0,m,m//world):
                    if plan.invoke(index,False,value[start:start+m//world],weight,local) or not torch.equal(local.view(torch.int16),reference[start:start+m//world].view(torch.int16)):good=False;break
                del reference
                if not good:break
            metadata['tried'].append(dict(attrs=attrs,exact=good))
            if good:plan.index=index;metadata['chosen']=attrs;return plan,metadata
        plan.close();return None,dict(metadata,reason='no bitwise-equivalent calibration')
    except Exception:plan.close();raise

class RowProjectionRouter:
    def __init__(self,original,group,*,empirical=False):
        if not isinstance(original,torch.nn.Linear) or original.bias is not None:raise TypeError('requires original ordinary bias-free Linear')
        self.original,self.group,self.empirical=original,group,empirical
        self.plans={};self.published=set();self.workspaces={}
        self.registration_index=0;self.registration_manifest=None;self.registration_ready=False;self.graph_bound=False
    def prepare_registration(self,shapes):
        """All ranks validate the entire startup sequence before any GPU work."""
        try:local=dict(shapes=[tuple(s) for s in shapes],error=None)
        except Exception as error:local=dict(shapes=None,error=repr(error))
        declarations=[None]*self.group.world_size
        dist.all_gather_object(declarations,local,group=self.group.cpu_group)
        accepted=all(d['error'] is None and d['shapes']==declarations[0]['shapes'] for d in declarations)
        self.registration_manifest=declarations[0]['shapes'] if accepted else None
        self.registration_ready=accepted
        self.registration_index=0
        return accepted,declarations
    def register(self,examples,*,discover_fn=discover):
        plan=None;metadata={};key=None;shape=None
        try:
            examples=list(examples)
            if not self.registration_ready:raise ValueError('startup registration sequence was not collectively accepted')
            if not examples:raise ValueError('empty calibration examples')
            shape=tuple(examples[0].shape);key=(shape,_stream(examples[0]))
            if key in self.plans:raise ValueError('stream/shape already registered')
            if self.registration_manifest is not None:
                if self.registration_index>=len(self.registration_manifest) or self.registration_manifest[self.registration_index]!=shape:raise ValueError('registration sequence disagrees with startup manifest')
            declaration=dict(shape=shape,index=self.registration_index,error=None)
        except Exception as error:declaration=dict(shape=shape,index=self.registration_index,error=repr(error))
        declarations=[None]*self.group.world_size
        dist.all_gather_object(declarations,declaration,group=self.group.cpu_group)
        self.registration_index+=1
        if any(d['error'] is not None or d['shape']!=declarations[0]['shape'] or d['index']!=declarations[0]['index'] for d in declarations):
            return dict(accepted=False,reason='inconsistent registration request',ranks=declarations)
        try:plan,metadata=discover_fn([(x,self.original.weight) for x in examples],self.group.world_size)
        except Exception as error:metadata=dict(reason='local discovery error',error=repr(error))
        # Even an unexpected local discovery failure joins this CPU collective.
        # Peers therefore fall back together instead of entering a GPU gather.
        accepted=torch.tensor(int(plan is not None),dtype=torch.int32)
        dist.all_reduce(accepted,op=dist.ReduceOp.MIN,group=self.group.cpu_group)
        if not int(accepted):
            if plan:plan.close()
            plan=None
        metadata=dict(metadata,accepted=bool(accepted))
        self.plans[key]=plan
        if plan:
            # Graphs for different shapes replay sequentially on one stream.
            # Keep one workspace per stream instead of one per capture bucket.
            # Different streams never alias scratch.
            if hasattr(plan,'scratch'):
                workspace_key=(plan.device,plan.stream,plan.scratch.numel())
                if workspace_key not in self.workspaces:self.workspaces[workspace_key]=plan.scratch
                plan.scratch=self.workspaces[workspace_key]
            self.published.add(shape)
        return metadata
    def __call__(self,x):
        shape=tuple(x.shape)
        if shape not in self.published:return self.original(x)
        if x.is_cuda and torch.cuda.is_current_stream_capturing():self.graph_bound=True
        plan=self.plans.get((shape,_stream(x)));native=None
        local=plan.local(x,self.original.weight,self.group.rank_in_group) if plan else None
        if local is None:
            native=self.original(x);count=x.shape[0]//self.group.world_size
            local=native.narrow(0,self.group.rank_in_group*count,count).contiguous()
        assembled=self.group.all_gather(local,dim=0)
        if self.empirical:return assembled
        if native is None:native=self.original(x)
        a,b=assembled.view(torch.int16),native.view(torch.int16)
        return torch.where((a==b).all(),a,b).view(torch.bfloat16)
    def close(self,*,graphs_destroyed=False):
        if self.graph_bound and not graphs_destroyed:raise RuntimeError('destroy captured graphs before closing projection plans')
        if self.graph_bound:torch.cuda.synchronize(self.original.weight.device)
        for plan in self.plans.values():
            if plan:plan.close()
        self.plans.clear();self.published.clear();self.workspaces.clear()
        self.graph_bound=False
        self.registration_ready=False
