"""Native CPU audit: execute exact wrapper AST with spies, index/reduction models."""
import ast,copy,hashlib,json,random,struct
from pathlib import Path
from types import SimpleNamespace as NS
P=Path(__file__).resolve().parent
class Tensor:
 def __init__(self,shape=(512,4096),dtype='bf16',device='cuda:0',ptr=4096,contiguous=True):
  self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=device;self.ptr=ptr;self.contiguous=contiguous;self.is_cuda=True
 def is_contiguous(self):return self.contiguous
 def data_ptr(self):return self.ptr
 def numel(self):
  n=1
  for s in self.shape:n*=s
  return n
 def element_size(self):return 2 if self.dtype=='bf16' else 4
 def t(self):return Tensor(self.shape[::-1],self.dtype,self.device)
 def to(self,dtype):events.append('cast');return Tensor(self.shape,dtype,self.device)
class Comm:
 def __init__(self):
  self.disabled=False;self.override_algo=None;self.max_pull_size=16*1024**2;self.graph=False;self.config=NS(algo='2shot_pull',use_graph=False,use_multicast=True);self.obj='comm'
 def _pick_config(self,nbytes,graph):
  events.append('policy');return self.config if nbytes<=self.max_pull_size else None
 def _can_use_graph(self):return self.graph
 def _allocate_graph_row(self,x,nbytes):events.append(('register',x.ptr,nbytes));return 'graph_row'
class Group:
 def __init__(self):self.world_size=8;self.ca_comm=Comm();self.flash=False;self.resolved='ca';self.rank_in_group=0
 def _can_use_flashinfer_allreduce(self,x):return self.flash
 def _resolve_outplace_all_reduce_method(self,x):return self.resolved
 def all_gather(self,x,dim):events.append(('gather',x.shape,dim));return Tensor((x.shape[0]*self.world_size,x.shape[1]))
def mm(a,b,out_dtype):events.append(('mm',a.shape,b.shape,out_dtype));return Tensor((a.shape[0],b.shape[1]),out_dtype)
def run(comm,x,out,graph,mc):events.append(('kernel',x.shape,out.shape,graph,mc))
torch=NS(bfloat16='bf16',float32='fp32',compiler=NS(is_compiling=lambda:state['compile']),cuda=NS(get_device_capability=lambda device:state['arch']),empty=lambda shape,**kw:Tensor(shape,**kw),mm=mm)
state={};events=[]
ns={'torch':torch,'envs':NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:state['deterministic'])),'GroupCoordinator':Group,'CustomAllReduceV2':Comm,'AllReduceAlgo':NS(TWO_SHOT_PULL='2shot_pull'),'module':lambda world:NS(run=run)}
tree=ast.parse((P/'reduce_unpool.py').read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('selected_policy','try_reduce_unpool')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exact_wrapper','exec'),ns)
def setup():
 state.clear();state.update(compile=False,arch=(9,0),deterministic=False);events.clear();return Group(),Tensor(),Tensor((8192,4096))
checks=[]
def check(label,cond):assert cond,label;checks.append(label)
mutations={
 'disabled_optin':lambda g,x,w,k:k.update(experimental=False),
 'model_ineligible':lambda g,x,w,k:k.update(model_eligible=False),
 'compile':lambda g,x,w,k:state.update(compile=True),
 'deterministic':lambda g,x,w,k:state.update(deterministic=True),
 'other_arch':lambda g,x,w,k:state.update(arch=(10,3)),
 'world3':lambda g,x,w,k:setattr(g,'world_size',3),
 'disabled_comm':lambda g,x,w,k:setattr(g.ca_comm,'disabled',True),
 'override':lambda g,x,w,k:setattr(g.ca_comm,'override_algo','2shot_pull'),
 'cpu':lambda g,x,w,k:setattr(x,'is_cuda',False),
 'rank3':lambda g,x,w,k:setattr(x,'ndim',3),
 'fp32':lambda g,x,w,k:setattr(x,'dtype','fp32'),
 'weight_dtype':lambda g,x,w,k:setattr(w,'dtype','fp32'),
 'weight_device':lambda g,x,w,k:setattr(w,'device','cuda:1'),
 'strided_input':lambda g,x,w,k:setattr(x,'contiguous',False),
 'strided_weight':lambda g,x,w,k:setattr(w,'contiguous',False),
 'small':lambda g,x,w,k:setattr(x,'shape',(128,4096)),
 'tail':lambda g,x,w,k:setattr(x,'shape',(513,4096)),
 'K_mismatch':lambda g,x,w,k:setattr(w,'shape',(8192,2048)),
 'input_ptr':lambda g,x,w,k:setattr(x,'ptr',4100),
 'weight_ptr':lambda g,x,w,k:setattr(w,'ptr',4100),
 'flashinfer':lambda g,x,w,k:setattr(g,'flash',True),
 'nccl':lambda g,x,w,k:setattr(g,'resolved',None),
 'mscclpp':lambda g,x,w,k:setattr(g,'resolved','pymscclpp'),
 'one_shot':lambda g,x,w,k:setattr(g.ca_comm.config,'algo','1shot_pull'),
 'graph_mc':lambda g,x,w,k:setattr(g.ca_comm.config,'use_graph',True),
 'above_cap':lambda g,x,w,k:setattr(x,'shape',(4096,4096)),
 'unaligned_width':lambda g,x,w,k:(setattr(x,'shape',(512,4100)),setattr(w,'shape',(8192,4100))),
}
for label,mut in mutations.items():
 g,x,w=setup();kw=dict(model_eligible=True,experimental=True);mut(g,x,w,kw)
 before=x.__dict__.copy();result=ns['try_reduce_unpool'](g,x,w,**kw)
 check(label,result is None and x.__dict__==before and not any(isinstance(e,tuple) and e[0] in ('register','kernel','mm','gather') for e in events))
for world in (2,4,8):
 for graph,mc in ((False,False),(False,True),(True,False)):
  g,x,w=setup();g.world_size=world;g.ca_comm.config=NS(algo='2shot_pull',use_graph=graph,use_multicast=mc);g.ca_comm.graph=graph
  before=x.__dict__.copy();result=ns['try_reduce_unpool'](g,x,w,model_eligible=True,experimental=True)
  check('eligible_%s_%s_%s'%(world,graph,mc),result.shape==(512,8192) and x.__dict__==before and ('mm',(512//world,4096),(4096,8192),'fp32') in events and sum(isinstance(e,tuple) and e[0]=='register' for e in events)==int(graph))
# Index mapping is separately checked from wrapper logic, including tile-loop tails.
for world in (2,4,8):
 for m in (512,520,1024,2048):
  for k in (1024,4096,8192):
   total=m*k//8;avg=total//world
   bounds=[(r*avg,(r+1)*avg) for r in range(world)]
   check('indices_%s_%s_%s'%(world,m,k),bounds[0][0]==0 and bounds[-1][1]==total and all(a*8==r*(m//world)*k and b*8==(r+1)*(m//world)*k for r,(a,b) in enumerate(bounds)))
# Demonstrate why a generic reduction replacement is rejected, even for BF16 inputs.
f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
def ordered(xs):
 a=xs[0]
 for x in xs[1:]:a=f32(a+x)
 return a
xs=[float(2**24),1.0,-float(2**24),0.0]
check('reduction_order_counterexample',ordered(xs)==0.0 and ordered([xs[0],xs[2],xs[1],xs[3]])==1.0)
# Kernel transformation proof is independent of the generated manifest.
stock=(P.parents[3]/'../..'/'sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh').resolve()
# Locate workspace without relying on current working directory.
stock=next(a/'sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh' for a in P.parents if (a/'sglang').is_dir())
s=stock.read_text(encoding='utf-8');a=s.index('template <typename Impl, typename T, uint32_t kWorldSize, bool kUsePDL>\nALL_REDUCE_KERNEL void all_reduce_2shot_pull_kernel');b=s.index('\ntemplate <uint32_t N>',a)
c=(P/'reduce_scatter_rows.cuh').read_text();ka=c.index('template <typename Impl');kb=c.index('\ntemplate <typename T, uint32_t',ka)
check('exact_kernel_transformation',c[ka:kb].strip().replace('muse_reduce_scatter_rows_kernel','all_reduce_2shot_pull_kernel').replace('vec.store(params.output, vid);','impl.store_multi(vec, vid);')==s[a:b].strip())
report={'status':'PASS','checks':len(checks),'check_names':checks,'scope':'CPU spy dispatch, complete-row index mapping, source transformation and FP32 order counterexample only. No CUDA, barrier liveness, IPC or multimem numerical proof.','source_sha256':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ('reduce_unpool.py','reduce_scatter_rows.cuh')}}
(P/'CPU_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','checks':len(checks)}))
