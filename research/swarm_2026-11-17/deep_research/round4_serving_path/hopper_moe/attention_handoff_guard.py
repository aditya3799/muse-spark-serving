"""Default-off research caller guard. Not imported by serving code.

Only for the audited serial Muse attention o_proj -> norm callsite. The lock
excludes users of THIS helper, not arbitrary stock communicator calls. All
ranks must complete collective admission before capture. Whole-worker foreign
stream ownership is still unvalidated; this is not a globally safe comm API.
"""
import threading
_LOCK=threading.Lock()
_ACTIVE=set()
_STREAM_OWNER={}

class PreparedAttentionHandoff:
 def __init__(self,comm,x,weight,stream,norm_kwargs,module,direct,device,stream_id,plane):
  self.comm,self.x,self.weight,self.residual=comm,x,weight,stream
  self.kw,self.module,self.direct=norm_kwargs,module,direct
  self.device,self.stream_id,self.plane=device,stream_id,plane
  self.identity=(id(comm.obj),comm._symm_tensor.data_ptr(),comm.max_push_size,comm.max_pull_size)
  self.operands=tuple((t.data_ptr(),tuple(t.shape),t.dtype) for t in [x,weight,stream,norm_kwargs['pre_weight'],norm_kwargs['alpha'],norm_kwargs['beta']])+(norm_kwargs['post_eps'],norm_kwargs['pre_eps'])
 def _live(self):
  fresh,reason=self._revalidate()
  assert fresh is not None,reason
  assert fresh.operands==self.operands,'operand layout/pointer or norm constants changed'
  import torch
  from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
  from sglang.srt.runtime_context import get_parallel
  g=get_parallel().attn_tp_group
  assert g.ca_comm is self.comm and g.world_size==8
  assert all(getattr(g,k,None) is None for k in ['hpu_communicator','xpu_communicator','pymscclpp_comm','_fi_workspace_hint'])
  c=self.comm;m=self.x.shape[0];nbytes=m*8192*2
  assert torch.cuda.current_device()==self.device
  assert torch.cuda.current_stream().cuda_stream==self.stream_id
  assert (id(c.obj),c._symm_tensor.data_ptr(),c.max_push_size,c.max_pull_size)==self.identity,'plane resized/replaced'
  cfg=c._pick_config(nbytes,c._can_use_graph())
  assert c.override_algo is None and cfg is not None and cfg.algo==AllReduceAlgo.TWO_SHOT_PULL and cfg.use_multicast and not cfg.use_graph
  assert self.plane.data_ptr()==self.module.local_address(c.obj),'not current local ordinaryVA plane'
 def run(self):
  import torch
  import torch.nn.functional as F
  self._live();key=id(self.comm)
  with _LOCK:
   assert key not in _ACTIVE,'reentrant helper plane use'
   assert _STREAM_OWNER.get(key,self.stream_id)==self.stream_id,'helper attempted another stream'
   _STREAM_OWNER[key]=self.stream_id;_ACTIVE.add(key)
  try:
   # Both output buffers are independent of the borrowed comm plane.
   out=torch.empty((self.x.shape[0],8192),dtype=torch.bfloat16,device=self.x.device)
   new_stream=torch.empty_like(self.residual)
   if self.direct:
    torch.mm(self.x,self.weight.t(),out=self.plane)
    branch=self.module.direct(self.comm.obj,self.plane,self.plane)
   else:
    partial=F.linear(self.x,self.weight,None)
    branch=self.module.staged(self.comm.obj,partial,self.plane)
   branch=branch if isinstance(branch,torch.Tensor) else torch.from_dlpack(branch)
   assert branch.data_ptr()==self.plane.data_ptr()
   fn=self.module.norm_small if self.x.shape[0]<192 else self.module.norm_large
   # Norm entry points are the original body with PDL=false. Enqueue before
   # releasing helper lease; subsequent same-stream plane writers wait for it.
   fn(branch,self.residual,out,new_stream,None,None,self.kw['pre_weight'],self.kw['alpha'],self.kw['beta'],self.kw['post_eps'],self.kw['pre_eps'],False)
   return out,new_stream,None
  finally:
   with _LOCK:_ACTIVE.remove(key)

def prepare_attention_handoff(comm,x,projection,residual,boundary,module,*,enabled=False,direct=False,collective_admission_complete=False,exclusive_serial_attention_callsite=False,skip_all_reduce=False,output_tensor=None):
 """Return (plan,reason), rejecting before caller skips the original AR.

 module must be the source-pinned h200_mc_gate module, precompiled before
 capture. There is no automatic module compilation or communicator allocation.
 The two explicit admission booleans are caller obligations, not inferred
 guarantees: all ranks and all foreign communicator users must be audited.
 """
 if not enabled:return None,'disabled'
 if not collective_admission_complete or not exclusive_serial_attention_callsite:return None,'collective/exclusive ownership contract absent'
 import torch
 if not x.is_cuda or torch.cuda.get_device_capability(x.device)!=(9,0) or 'H200' not in torch.cuda.get_device_name(x.device).upper():return None,'not qualified H200 architecture'
 from sglang.srt.runtime_context import get_parallel,get_forward,get_exec,get_disagg
 from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
 from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
 from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend,UnquantizedLinearMethod
 from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
 from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
 if skip_all_reduce or output_tensor is not None or should_skip_mlp_all_reduce():return None,'linear callsite reduction/output override'
 if torch.compiler.is_compiling() or is_batch_invariant_mode_enabled():return None,'compiler/batch-invariant override'
 if get_exec().overlap.enable_two_batch_overlap or get_disagg().enable_pdmux:return None,'multistream model execution unsupported'
 parallel=get_parallel()
 if parallel.tp_size!=8 or parallel.attn_tp_size!=8 or parallel.attn_cp_size!=1 or parallel.attn_dp_enabled or getattr(get_forward(),'sp_active',False):return None,'parallel layout unsupported'
 group=parallel.attn_tp_group
 if getattr(group,'ca_comm',None) is not comm:return None,'wrong attention communicator'
 if group.world_size!=8 or any(getattr(group,k,None) is not None for k in ['hpu_communicator','xpu_communicator','pymscclpp_comm','_fi_workspace_hint']):return None,'higher-priority group collective path present'
 if type(comm) is not CustomAllReduceV2 or comm.disabled or not comm.has_multicast or not comm.pull_enabled or comm.world_size!=8:return None,'V2 multicast plane unavailable'
 if comm.override_algo is not None:return None,'algorithm override'
 if get_bf16_gemm_backend().value not in ['auto','torch'] or type(projection.quant_method) is not UnquantizedLinearMethod:return None,'projection backend not qualified'
 if torch.are_deterministic_algorithms_enabled() or get_exec().deterministic.enable_deterministic_inference:return None,'deterministic override unsupported'
 if get_exec().comm.enable_quant_communications:return None,'quantized communication unsupported'
 if x.ndim!=2 or x.shape[1]!=1024 or not 32<=x.shape[0]<=1024:return None,'shape unsupported'
 m=x.shape[0];w=projection.weight;nbytes=m*8192*2
 if projection.bias is not None or not projection.input_is_parallel or not projection.reduce_results or not projection.use_dp_attention_reduce or projection.use_decode_attn_tp or projection.tp_group is not parallel.attn_tp_group:return None,'projection contract differs'
 if x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or tuple(w.shape)!=(8192,1024) or w.requires_grad or x.requires_grad:return None,'projection type/shape/gradient unsupported'
 if not x.is_contiguous() or not w.is_contiguous() or x.data_ptr()%16 or w.data_ptr()%16:return None,'projection layout unsupported'
 if residual.dtype!=torch.float32 or tuple(residual.shape)!=(m,8192) or not residual.is_contiguous():return None,'residual layout unsupported'
 device=torch.cuda.current_device()
 if x.device.index!=device or w.device!=x.device or residual.device!=x.device or comm.device.index!=device:return None,'device mismatch'
 if comm.config.num_mc_blocks!=16 or nbytes>comm.max_pull_size or nbytes>16*1024*1024:return None,'current plane capacity/geometry mismatch'
 cfg=comm._pick_config(nbytes,comm._can_use_graph())
 if cfg is None or cfg.algo!=AllReduceAlgo.TWO_SHOT_PULL or not cfg.use_multicast or cfg.use_graph:return None,'selected baseline algorithm is not staged multicast'
 if not boundary.jit_selected() or boundary._hidden_size!=8192 or boundary._post_norm.has_weight or not boundary._pre_norm.has_weight or boundary._post_norm_is_input_scale or not boundary._use_residual_gate:return None,'norm backend/specialization unsupported'
 if not getattr(boundary,'_weights_cached',False):return None,'norm cache must be prepared before admission'
 kw=boundary.jit_kernel_kwargs(torch.bfloat16)
 if kw['post_weight'] is not None or any(kw[k] is None for k in ['pre_weight','alpha','beta']):return None,'norm parameters absent'
 for name,dtype in [('pre_weight',torch.bfloat16),('alpha',torch.float32),('beta',torch.float32)]:
  t=kw[name]
  if t.dtype!=dtype or tuple(t.shape)!=(8192,) or not t.is_contiguous() or t.device!=x.device:return None,'norm parameter layout unsupported'
 offset=16*comm.max_push_size;slab=comm._symm_tensor.view(torch.uint8)
 if offset+nbytes>slab.numel():return None,'local slab span exceeds allocation'
 plane=slab[offset:offset+nbytes].view(torch.bfloat16).view(m,8192)
 if plane.data_ptr()!=module.local_address(comm.obj):return None,'local pointer differs from current comm object'
 lo,hi=plane.data_ptr(),plane.data_ptr()+nbytes
 for t in [x,w,residual,kw['pre_weight'],kw['alpha'],kw['beta']]:
  if not (t.data_ptr()+t.numel()*t.element_size()<=lo or hi<=t.data_ptr()):return None,'live operand aliases plane'
 stream_id=torch.cuda.current_stream().cuda_stream
 with _LOCK:
  if id(comm) in _ACTIVE or _STREAM_OWNER.get(id(comm),stream_id)!=stream_id:return None,'helper lease or stream conflict'
 plan=PreparedAttentionHandoff(comm,x,w,residual,kw,module,direct,device,stream_id,plane)
 plan._revalidate=lambda:prepare_attention_handoff(comm,x,projection,residual,boundary,module,enabled=enabled,direct=direct,collective_admission_complete=collective_admission_complete,exclusive_serial_attention_callsite=exclusive_serial_attention_callsite,skip_all_reduce=skip_all_reduce,output_tensor=output_tensor)
 return plan,'eligible research-only serial callsite'
