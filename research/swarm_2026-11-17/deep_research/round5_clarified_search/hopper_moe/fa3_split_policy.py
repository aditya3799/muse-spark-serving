"""Default-off per-target FA3 split policy. No precision/kernel changes.

Install only by setting model_runner._research_fa3_split_policy before creating a
NEW attention backend. Never attach to a captured/shared backend. An immutable
policy is tied to that backend's lifetime; failed capture poisons it. All-rank
installation agreement and native binary precision qualification remain callers'
obligations. FullCudaGraphBackend integration rejects unsupported lifecycle paths.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import threading

class SplitPolicyError(RuntimeError): pass

@dataclass(frozen=True)
class TargetFA3SplitPolicy:
    verify: int = 0
    decode: int = 0
    enabled: bool = False
    # These must be established outside capture, coherently on every rank.
    collective_setup_agreed: bool = False
    precision_binary_audited: bool = False
    def __post_init__(self):
        for x in (self.verify,self.decode):
            if type(x) is not int or x not in (0,1,2,4,8):
                raise SplitPolicyError('initial bounded split set is 0,1,2,4,8')

def phase_of(batch):
    mode=batch.forward_mode
    if mode.is_target_verify(): return 'verify'
    if mode.is_decode_or_idle(): return 'decode'
    if mode.is_draft_extend_v2(): raise SplitPolicyError('draft mode on target split backend')
    return 'other'

def signature(t):
    if t is None:return None
    return (id(t),int(t.data_ptr()),tuple(t.shape),str(t.dtype),tuple(t.stride()))

def tensor_span(t):
    if t is None:return None
    return (int(t.data_ptr()),tuple(t.shape),str(t.dtype),tuple(t.stride()),str(getattr(t,"device","cpu")))

def record_scheduler_copy(backend,source,buffer,n):
    c=controller(backend)
    if c is None:return
    c.check()
    if c.last_scheduler is None or c.last_scheduler[2] is not source or tuple(source.shape)!=(n,) or tuple(source.stride())!=(1,) or len(buffer.shape)!=1 or buffer.shape[0]<n or tuple(buffer.stride())!=(1,):
        raise SplitPolicyError("scheduler copy has wrong source/span")
    if str(source.dtype)!=str(buffer.dtype) or str(getattr(source,"device","cpu"))!=str(getattr(buffer,"device","cpu")):
        raise SplitPolicyError("scheduler copy dtype/device mismatch")
    c.scheduler_copy=(c.scheduler_serial,(int(buffer.data_ptr()),(n,),str(buffer.dtype),(1,),str(getattr(buffer,"device","cpu"))))

def metadata_signature(md):
    if md is None:raise SplitPolicyError('missing attention metadata')
    return (id(md),signature(getattr(md,'scheduler_metadata',None)))

class SplitController:
    """Pure host state machine; CPU tests instantiate it with source fixtures."""
    def __init__(self,backend,owner,policy):
        self.backend,self.owner,self.policy=backend,owner,policy
        self.original=(backend.num_splits,backend.decode_num_splits)
        self.fingerprint=(policy.verify,policy.decode,policy.enabled)
        self.phase=None;self.depth=0;self.thread=None;self.poison=None
        self.metadata={};self.graphs={};self.capture_depth=0
        self.last_metadata_phase=None
        self.scheduler_serial=0;self.last_scheduler=None;self.scheduler_copy=None
        self.runtime_snapshot=None;self.runtime_expected=None
    def check(self):
        if self.poison:raise SplitPolicyError('backend poisoned: '+self.poison)
        if self.runtime_snapshot is not None and self.runtime_snapshot()!=self.runtime_expected:
            raise SplitPolicyError('runtime admission changed; invalidate backend and captured graphs')
        if getattr(self.owner,'_research_fa3_split_policy',None) is not self.policy:
            raise SplitPolicyError('policy replaced; create fresh backend and graphs')
        if (self.policy.verify,self.policy.decode,self.policy.enabled)!=self.fingerprint:
            raise SplitPolicyError('policy mutated')
        if self.backend.is_draft_runner or self.backend.fa_impl_ver!=3:
            raise SplitPolicyError('backend role/version changed')
        expected=self.selected(self.phase) if self.depth else self.original
        if (self.backend.num_splits,self.backend.decode_num_splits)!=expected:
            raise SplitPolicyError('split fields changed outside policy scope')
    def selected(self,phase):
        if phase=='verify':return self.policy.verify,self.policy.verify
        if phase=='decode':return self.policy.decode,self.policy.decode
        return self.original
    def fail(self,reason):
        self.poison=reason;self.graphs.clear();self.metadata.clear()
    @contextmanager
    def scope(self,phase):
        self.check();outer=self.depth==0
        if not outer and (self.thread!=threading.get_ident() or self.phase!=phase):
            raise SplitPolicyError('concurrent or cross-phase backend use')
        if outer:
            self.phase=phase;self.thread=threading.get_ident()
            self.backend.num_splits,self.backend.decode_num_splits=self.selected(phase)
        self.depth+=1
        try:yield
        finally:
            self.depth-=1
            if outer:
                self.backend.num_splits,self.backend.decode_num_splits=self.original
                self.phase=None;self.thread=None
    def stamp(self,phase,before_serial):
        md=self.backend.forward_metadata
        if getattr(md,"scheduler_metadata",None) is not None:
            if self.scheduler_serial==before_serial:
                self.require_metadata(phase)  # Only an already-qualified no-op refresh may reuse it.
            elif self.last_scheduler is None or self.last_scheduler[0]!=phase or self.last_scheduler[1]!=self.selected(phase) or self.last_scheduler[2] is None:
                raise SplitPolicyError("scheduler metadata lacks a coherent producer")
            elif tensor_span(md.scheduler_metadata)!=tensor_span(self.last_scheduler[2]) and self.scheduler_copy!=(self.scheduler_serial,tensor_span(md.scheduler_metadata)):
                raise SplitPolicyError("scheduler metadata is not producer result or proven copied buffer prefix")
        self.metadata[id(md)]=(md,phase,self.selected(phase),metadata_signature(md))
        self.last_metadata_phase=phase
    def require_metadata(self,phase):
        md=self.backend.forward_metadata;record=self.metadata.get(id(md))
        if record is None or record[0] is not md or record[1]!=phase or record[2]!=self.selected(phase) or record[3]!=metadata_signature(md):
            raise SplitPolicyError('missing/stale/cross-phase scheduler metadata stamp')
    def runner_guard(self,runner):
        self.check()
        if runner.model_runner is not self.owner or runner._replay_attn_backend() is not self.backend:
            raise SplitPolicyError('runner/backend ownership mismatch')
        if self.owner.is_draft_worker or runner.enable_pdmux or runner.enable_two_batch_overlap or runner.ragged_verify_mode:
            raise SplitPolicyError('draft/PDMux/TBO/ragged graphs unsupported initially')
        if getattr(runner,'_metadata_glue',None) is not None:
            raise SplitPolicyError('metadata glue lifecycle is not admitted')
        if type(runner.backend).__name__!='FullCudaGraphBackend':
            raise SplitPolicyError('only pinned full CUDA graph lifecycle admitted')
    def graph_valid(self,gb,key):
        try:self.runner_guard(gb._cuda_graph_runner)
        except SplitPolicyError:return False
        record=self.graphs.get((id(gb),key))
        return record is not None and gb._graphs.get(key) is record['graph']
    def require_graph(self,gb,key,batch):
        if not self.graph_valid(gb,key):raise SplitPolicyError('uncaptured/stale/replaced/poisoned graph')
        record=self.graphs[(id(gb),key)];phase=phase_of(batch)
        if record['phase']!=phase or record['selected']!=self.selected(phase):
            raise SplitPolicyError('graph split/forward-mode mismatch')
        self.require_metadata(phase)
        if id(self.backend.forward_metadata)!=record['metadata_id']:
            raise SplitPolicyError('replay did not select graph-owned metadata')

def controller(backend):return getattr(backend,'_research_fa3_controller',None)

def initialize_target_fa3_split_policy(backend,owner):
    p=getattr(owner,'_research_fa3_split_policy',None)
    if p is None or not p.enabled:return
    if type(p) is not TargetFA3SplitPolicy:raise SplitPolicyError('wrong policy type')
    if not p.collective_setup_agreed or not p.precision_binary_audited:
        raise SplitPolicyError('all-rank setup/native precision audit absent')
    if owner.is_draft_worker or backend.is_draft_runner or backend.fa_impl_ver!=3:
        raise SplitPolicyError('target FA3 only; draft FA4 stays unchanged')
    if backend.use_mla or backend.is_encoder_decoder or backend.topk>1 or backend.kv_cache_is_mxfp8:
        raise SplitPolicyError('MLA/cross-attention/cascade/MXFP8 unsupported')
    if str(backend.kv_cache_dtype)!='torch.float8_e4m3fn':raise SplitPolicyError('FP8 E4M3 target KV required')
    if (backend.head_dim,backend.num_attention_heads,backend.num_kv_heads)!=(64,16,2):
        raise SplitPolicyError('only audited target TP8 head geometry')
    if backend.decode_cuda_graph_metadata or backend.target_verify_metadata or backend.forward_metadata is not None:
        raise SplitPolicyError('install before any metadata/capture')
    from sglang.srt.runtime_context import get_parallel,get_exec,get_disagg
    import torch
    parallel=get_parallel()
    if parallel.attn_tp_size!=8 or parallel.attn_dp_enabled or parallel.attn_cp_size!=1:
        raise SplitPolicyError('TP8 without DP/CP only')
    if get_exec().deterministic.enable_deterministic_inference or get_exec().overlap.enable_two_batch_overlap or get_disagg().enable_pdmux:
        raise SplitPolicyError('deterministic/TBO/PDMux override unsupported')
    device=backend.device
    parsed_device=torch.device(device)
    if parsed_device.type!='cuda' or parsed_device.index not in (None,torch.cuda.current_device()):
        raise SplitPolicyError('backend must use current CUDA device')
    if torch.cuda.get_device_capability(device)!=(9,0) or 'H200' not in torch.cuda.get_device_name(device).upper():
        raise SplitPolicyError('native H200 gate required; no SM86 qualification')
    c=SplitController(backend,owner,p)
    def snapshot():
        pa=get_parallel();ex=get_exec();di=get_disagg()
        return (backend.use_mla,backend.is_encoder_decoder,backend.topk,backend.kv_cache_is_mxfp8,str(backend.kv_cache_dtype),backend.head_dim,backend.num_attention_heads,backend.num_kv_heads,str(backend.device),torch.cuda.current_device(),pa.attn_tp_size,pa.attn_dp_enabled,pa.attn_cp_size,ex.deterministic.enable_deterministic_inference,ex.overlap.enable_two_batch_overlap,di.enable_pdmux)
    c.runtime_snapshot=snapshot;c.runtime_expected=snapshot()
    backend._research_fa3_controller=c

def fa3_split_scope(kind):
    """Wrap real pinned metadata/forward methods, retaining their whole bodies."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(self,*args,**kwargs):
            c=controller(self)
            if c is None:return fn(self,*args,**kwargs)
            if kind=='metadata':batch=kwargs.get('forward_batch',args[0] if args else None)
            else:batch=kwargs.get('forward_batch',args[4] if len(args)>4 else None)
            if batch is None:raise SplitPolicyError('missing forward batch')
            phase=phase_of(batch)
            with c.scope(phase):
                if kind=='forward':c.require_metadata(phase)
                before_serial=c.scheduler_serial
                try:result=fn(self,*args,**kwargs)
                except Exception:
                    if kind=='metadata':c.fail('metadata creation/refresh failed')
                    raise
                c.check()
                if kind=='metadata':c.stamp(phase,before_serial)
                return result
        return wrapped
    return decorate

def fa3_scheduler_guard(fn):
    @wraps(fn)
    def wrapped(self,*args,**kwargs):
        c=controller(self)
        if c is not None:
            c.check()
            if c.depth==0:raise SplitPolicyError('scheduler generation outside stamped phase')
        result=fn(self,*args,**kwargs)
        if c is not None:
            c.scheduler_serial+=1
            c.last_scheduler=(c.phase,c.selected(c.phase),result)
        return result
    return wrapped

def _runner_controller(runner):
    resolve=getattr(runner,'_replay_attn_backend',None)
    if resolve is None:
        policy=getattr(getattr(runner,'model_runner',None),'_research_fa3_split_policy',None)
        if policy is not None and getattr(policy,'enabled',False):
            raise SplitPolicyError('enabled policy requires reviewed decode/verify graph runner')
        return None
    return controller(resolve())

def fa3_runner_capture_guard(fn):
    @wraps(fn)
    def wrapped(self,*args,**kwargs):
        c=_runner_controller(self)
        if c is None:return fn(self,*args,**kwargs)
        c.runner_guard(self)
        try:return fn(self,*args,**kwargs)
        except Exception:
            c.fail('capture_one_shape failed; fresh backend/runner required')
            raise
    return wrapped

def fa3_graph_capture_guard(fn):
    @wraps(fn)
    def wrapped(self,key,*args,**kwargs):
        c=_runner_controller(self._cuda_graph_runner)
        if c is None:return fn(self,key,*args,**kwargs)
        c.runner_guard(self._cuda_graph_runner)
        phase=c.last_metadata_phase
        if phase not in ('verify','decode'):raise SplitPolicyError('split graph mode unsupported')
        c.require_metadata(phase)
        c.graphs.pop((id(self),key),None);c.capture_depth+=1
        try:
            result=fn(self,key,*args,**kwargs)
            c.check();c.require_metadata(phase)
            graph=self._graphs.get(key)
            if graph is None:raise SplitPolicyError('capture returned without published graph')
            c.graphs[(id(self),key)]={'graph':graph,'phase':phase,'selected':c.selected(phase),'metadata_id':id(c.backend.forward_metadata)}
            return result
        except Exception:
            c.fail('native graph capture failed; fresh backend/runner required')
            raise
        finally:c.capture_depth-=1
    return wrapped

def fa3_graph_replay_guard(fn):
    @wraps(fn)
    def wrapped(self,key,batch,*args,**kwargs):
        c=_runner_controller(self._cuda_graph_runner)
        if c is not None:c.require_graph(self,key,batch)
        return fn(self,key,batch,*args,**kwargs)
    return wrapped

def fa3_graph_can_run_guard(fn):
    @wraps(fn)
    def wrapped(self,batch,key,*args,**kwargs):
        c=_runner_controller(self._cuda_graph_runner)
        if c is not None and not c.graph_valid(self,key):return False
        return fn(self,batch,key,*args,**kwargs)
    return wrapped

def fa3_graph_cleanup_guard(fn):
    @wraps(fn)
    def wrapped(self,*args,**kwargs):
        c=_runner_controller(self._cuda_graph_runner)
        if c is not None:
            c.graphs={k:v for k,v in c.graphs.items() if k[0]!=id(self)}
        return fn(self,*args,**kwargs)
    return wrapped
