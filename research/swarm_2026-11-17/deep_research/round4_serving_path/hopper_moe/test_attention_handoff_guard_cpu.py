"""CPU-only control-flow tests; mocked tensors are not CUDA/MC evidence."""
import importlib.util,json,hashlib,sys,types,pathlib,math
P=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('guard',P/'attention_handoff_guard.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
N=types.SimpleNamespace
class Dev:
 def __init__(self,index=0):self.index=index
 def __eq__(self,o):return isinstance(o,Dev) and o.index==self.index
class Tensor:
 def __init__(self,shape,dtype='bf16',ptr=100000000,device=None):self.shape=shape;self.dtype=dtype;self.ptr=ptr;self.device=device or Dev();self.requires_grad=False;self.is_cuda=True;self.contiguous=True
 @property
 def ndim(self):return len(self.shape)
 def numel(self):
  n=1
  for x in self.shape:n*=x
  return n
 def element_size(self):return {'bf16':2,'fp32':4,'u8':1}[self.dtype]
 def data_ptr(self):return self.ptr
 def is_contiguous(self):return self.contiguous
 def view(self,*args):
  if len(args)==1 and isinstance(args[0],str):
   dtype=args[0];return Tensor((self.numel()*self.element_size()//{'bf16':2,'fp32':4,'u8':1}[dtype],),dtype,self.ptr,self.device)
  return Tensor(tuple(args),self.dtype,self.ptr,self.device)
 def __getitem__(self,s):return Tensor((s.stop-s.start,),self.dtype,self.ptr+s.start*self.element_size(),self.device)
class CustomAllReduceV2:pass
class UnquantizedLinearMethod:pass
class Algo:TWO_SHOT_PULL='2shot'
def mod(name,**kw):
 m=types.ModuleType(name);m.__dict__.update(kw);sys.modules[name]=m
 return m

def fixture():
 state=N(cap=(9,0),name='NVIDIA H200',current=0,stream=11,compiling=False,batch=False,skip=False,backend='auto',det=False)
 cuda=N(get_device_capability=lambda d=None:state.cap,get_device_name=lambda d=None:state.name,current_device=lambda:state.current,current_stream=lambda:N(cuda_stream=state.stream))
 mod('torch',cuda=cuda,bfloat16='bf16',float32='fp32',uint8='u8',compiler=N(is_compiling=lambda:state.compiling),are_deterministic_algorithms_enabled=lambda:state.det)
 c=CustomAllReduceV2();c.disabled=False;c.has_multicast=True;c.pull_enabled=True;c.world_size=8;c.override_algo=None;c.device=Dev();c.config=N(num_mc_blocks=16);c.max_push_size=1048576;c.max_pull_size=16777216;c.obj=object();c._symm_tensor=Tensor((33562624,),'u8',1000000);c.cfg=N(algo='2shot',use_multicast=True,use_graph=False);c._pick_config=lambda n,g:c.cfg;c._can_use_graph=lambda:False
 group=N(ca_comm=c,world_size=8,hpu_communicator=None,xpu_communicator=None,pymscclpp_comm=None,_fi_workspace_hint=None)
 parallel=N(tp_size=8,attn_tp_size=8,attn_dp_enabled=False,attn_cp_size=1,attn_tp_group=group);forward=N(sp_active=False);execution=N(deterministic=N(enable_deterministic_inference=False),comm=N(enable_quant_communications=False),overlap=N(enable_two_batch_overlap=False));disagg=N(enable_pdmux=False)
 mod('sglang.srt.runtime_context',get_parallel=lambda:parallel,get_forward=lambda:forward,get_exec=lambda:execution,get_disagg=lambda:disagg)
 mod('sglang.srt.batch_invariant_ops',is_batch_invariant_mode_enabled=lambda:state.batch)
 mod('sglang.srt.layers.moe.utils',should_skip_mlp_all_reduce=lambda:state.skip)
 mod('sglang.srt.layers.quantization.unquant',get_bf16_gemm_backend=lambda:N(value=state.backend),UnquantizedLinearMethod=UnquantizedLinearMethod)
 mod('sglang.srt.distributed.device_communicators.custom_all_reduce_v2',CustomAllReduceV2=CustomAllReduceV2)
 mod('sglang.kernels.ops.communication.all_reduce',AllReduceAlgo=Algo)
 x=Tensor((32,1024),ptr=100000000);w=Tensor((8192,1024),ptr=110000000);r=Tensor((32,8192),'fp32',150000000)
 projection=N(weight=w,bias=None,input_is_parallel=True,reduce_results=True,use_dp_attention_reduce=True,use_decode_attn_tp=False,tp_group=group,quant_method=UnquantizedLinearMethod())
 kw=dict(post_weight=None,pre_weight=Tensor((8192,),ptr=160000000),alpha=Tensor((8192,),'fp32',170000000),beta=Tensor((8192,),'fp32',180000000),post_eps=1e-8,pre_eps=1e-5)
 boundary=N(jit_selected=lambda:True,_hidden_size=8192,_post_norm=N(has_weight=False),_pre_norm=N(has_weight=True),_post_norm_is_input_scale=False,_use_residual_gate=True,_weights_cached=True,jit_kernel_kwargs=lambda dtype:kw)
 module=N(local_address=lambda obj:c._symm_tensor.data_ptr()+16*c.max_push_size)
 opts=dict(enabled=True,collective_admission_complete=True,exclusive_serial_attention_callsite=True)
 return locals()
def invoke(f):return guard.prepare_attention_handoff(f['c'],f['x'],f['projection'],f['r'],f['boundary'],f['module'],**f['opts'])
checks=[]
f=fixture();plan,reason=invoke(f);assert plan is not None;plan._live();checks.append(dict(name='eligible and live',pass_=True))
mutations={
 'disabled':"opts['enabled']=False",'no collective consensus':"opts['collective_admission_complete']=False",'no exclusive ownership':"opts['exclusive_serial_attention_callsite']=False",
 'SM86':"state.cap=(8,6)",'H100':"state.name='H100'",'wrong current ordinal':"state.current=1",'wrong comm device':"c.device=Dev(1)",
 'CP':"parallel.attn_cp_size=2",'TBO':"execution.overlap.enable_two_batch_overlap=True",'PDMux':"disagg.enable_pdmux=True",'DP':"parallel.attn_dp_enabled=True",'SP':"forward.sp_active=True",'TP4':"parallel.tp_size=4",'wrong group':"group.ca_comm=None",'MSCCLPP precedence':"group.pymscclpp_comm=object()",'flashinfer hint':"group._fi_workspace_hint='attn_tp'",
 'comm disabled':"c.disabled=True",'no MC':"c.has_multicast=False",'no pull':"c.pull_enabled=False",'algo override':"c.override_algo='push'",'GEMV':"state.backend='gemv'",'batch invariant':"state.batch=True",'compiler':"state.compiling=True",'deterministic':"state.det=True",'quant comm':"execution.comm.enable_quant_communications=True",
 'M8 push scope':"x.shape=(8,1024)",'K mismatch':"x.shape=(32,4096)",'weight shape':"w.shape=(8192,512)",'grad':"w.requires_grad=True",'strided':"x.contiguous=False",'bias':"projection.bias=object()",'skip caller':"opts['skip_all_reduce']=True",'skip forward flag':"state.skip=True",'output override':"opts['output_tensor']=object()",'wrong reduction':"projection.use_dp_attention_reduce=False",
 'too small pull capacity':"c.max_pull_size=1024",'wrong MC blocks':"c.config.num_mc_blocks=8",'no selected algo':"c.cfg=None",'selected push':"c.cfg.algo='push'",'selected unicast':"c.cfg.use_multicast=False",'selected graph pointer':"c.cfg.use_graph=True",'post weight present':"boundary._post_norm.has_weight=True",'norm cache unprepared':"boundary._weights_cached=False",'2D gate':"kw['alpha'].shape=(1,8192)",'wrong gate dtype':"kw['beta'].dtype='bf16'",'wrong plane ptr':"module.local_address=lambda obj:0",'short slab':"c._symm_tensor.shape=(8,)",'live operand alias':"x.ptr=c._symm_tensor.ptr+16*c.max_push_size"
}
for name,code in mutations.items():
 f=fixture();exec(code,globals(),f);p,why=invoke(f);assert p is None,(name,why);checks.append(dict(name=name,pass_=True,rejection=why))
for name,code in {'resize':'c.max_pull_size+=16','replacement':'c.obj=object()','algorithm change':"c.cfg.algo='push'",'stream change':'state.stream+=1','foreign backend added':"group._fi_workspace_hint='attn_tp'",'operand pointer change':'w.ptr+=16','norm epsilon change':"kw['post_eps']=1e-6",'backend change':"state.backend='gemv'",'later skip flag':'state.skip=True'}.items():
 f=fixture();p,_=invoke(f);exec(code,globals(),f)
 try:p._live()
 except AssertionError:checks.append(dict(name='live '+name,pass_=True));continue
 raise AssertionError(name)
report=dict(status='CPU_MOCKED_ADMISSION_PASS',scope='Guard control flow only. No CUDA, real communicator import, foreign-stream ownership proof, or native runtime qualification.',count=len(checks),checks=checks,source_sha256=hashlib.sha256((P/'attention_handoff_guard.py').read_bytes()).hexdigest())
(P/'ATTENTION_HANDOFF_GUARD_CPU_TESTS.json').write_text(json.dumps(report,indent=2));print(report['status'],report['count'])
