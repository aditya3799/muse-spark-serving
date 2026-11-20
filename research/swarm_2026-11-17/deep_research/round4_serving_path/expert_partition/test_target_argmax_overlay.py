"""CPU lifecycle/admission tests. Fake tensors/events; no CUDA arithmetic claim."""
import ast, contextlib, hashlib, json
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import dataclass
H=Path(__file__).resolve().parent; O=H/'target_argmax_overlay/python/sglang/srt'
checks=[]
def ok(name,value):
    assert value,name; checks.append(name)
class Tensor:
    def __init__(self,n,root=None): self.n=n; self.root=root or self; self.shape=(n,); self.dtype='long'; self.device='cuda'; self.values=[0]*n
    def numel(self): return self.n
    def data_ptr(self): return id(self.root)
    def storage_offset(self): return 0
    def stride(self): return (1,)
    def untyped_storage(self): return NS(nbytes=lambda:self.root.n*8)
    def __getitem__(self,key): return Tensor(key.stop,self.root)
    def copy_(self,x): self.values=list(x.values); return self
cuda=NS(capturing=False,stream_key=101); cuda.current_stream=lambda device:NS(cuda_stream=cuda.stream_key); cuda.is_current_stream_capturing=lambda:cuda.capturing
torch=NS(cuda=cuda,long='long',empty=lambda n,**kw:Tensor(n),device=lambda x:NS(type=x))
core_ns={'torch':torch}
def load_source(path,ns):
    tree=ast.parse(path.read_text()); tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(tree,str(path),'exec'),ns)
load_source(O/'speculative/dflash_target_argmax_capture.py',core_ns)
parallel=NS(tp_size=8,tp_group=NS(ranks=list(range(8)),rank_in_group=0,world_size=8),attn_dp_enabled=False,attn_cp_size=1,attn_dcp_size=1,pp_size=1)
disagg=NS(enable_pdmux=False); execution=NS(overlap=NS(enable_two_batch_overlap=False))
ns={'torch':torch,'TargetArgmaxCapture':core_ns['TargetArgmaxCapture'],'get_parallel':lambda:parallel,'get_disagg':lambda:disagg,'get_exec':lambda:execution}
load_source(O/'speculative/dflash_target_argmax_adapter.py',ns)
@dataclass(frozen=True)
class Key:
    size:int=2
    stream_idx:object=None
    variant_label:object=None
    attention_variant:object=None
mode=NS(is_target_verify=lambda:True)
DFlashWorkerV2=type('DFlashWorkerV2',(),{}); DecodeCudaGraphRunner=type('DecodeCudaGraphRunner',(),{}); FullCudaGraphBackend=type('FullCudaGraphBackend',(),{})
def fixture():
    worker=DFlashWorkerV2(); worker.device='cuda'; worker.block_size=8; worker._is_domino=False
    lp=NS(final_logit_softcapping=30.0,vocab_size=256,defer_verify_gather=True,is_deferred_shard=lambda x:True,deferred_argmax=lambda x:Tensor(x.shape[0]))
    worker._target_logits_processor=lp
    mr=NS(forward_stream=NS(cuda_stream=101),is_draft_worker=False,model_config=NS(is_encoder_decoder=False),lora_manager=None,decode_cuda_graph_runner=None,capture_tail_hooks=[])
    worker.target_worker=NS(model_runner=mr)
    adapter=ns['maybe_install_target_argmax_capture'](worker,enabled=True)
    runner=DecodeCudaGraphRunner(); runner.model_runner=mr; runner.backend=FullCudaGraphBackend(); runner.backend._graphs={}
    runner.capture_forward_mode=mode; runner.captured_req_width=8; runner.max_bs=4; runner.bs=2; runner.raw_num_token=16
    for k in ['require_gathered_buffer','require_mlp_sync','enable_two_batch_overlap','enable_prefill_cp','enable_pdmux','record_nolora_graph','ragged_verify_mode','is_dllm']: setattr(runner,k,False)
    runner.attention_graph_variants=None; runner._replay_graph_key=Key(); mr.decode_cuda_graph_runner=runner
    return worker,mr,adapter,runner
fb=NS(forward_mode=mode); out=NS(next_token_logits=NS(shape=(16,256)))
def ready():
    w,m,a,r=fixture(); a.on_capture_start(r); cuda.capturing=True; a(r,out,fb,16); cuda.capturing=False; r.backend._graphs[Key()]=object(); a.on_capture_success(r,fb,16,Key()); return w,m,a,r
def consume(a,**kw):
    args=dict(raw_rows=16,graph_used=True,can_skip_verify_logits=True); args.update(kw); return a.consume(**args)
ok('default off touches no worker attributes',ns['maybe_install_target_argmax_capture'](object()) is None)
w,m,a,r=fixture(); del m.decode_cuda_graph_runner; ok('constructor before runner attribute exists',ns['maybe_install_target_argmax_capture'](w,enabled=True) is not None); m.decode_cuda_graph_runner=r; m.capture_tail_hooks=[a]; ok('registered before target runner capture',m.capture_tail_hooks==[a]); a.on_capture_start(r); a(r,out,fb,16); r.backend._graphs[Key()]=object(); a.on_capture_success(r,fb,16,Key()); ok('warmup cannot qualify',consume(a) is None)
w,m,a,r=ready(); ok('successful actual capture qualifies',consume(a).numel()==16)
for arg,val in [('graph_used',False),('can_skip_verify_logits',False),('raw_rows',8),('raw_rows',24)]: ok('reject '+arg+str(val),consume(a,**{arg:val}) is None)
for field,value in [('final_logit_softcapping',31.0),('vocab_size',255),('defer_verify_gather',False)]:
    lp=w._target_logits_processor; oldvalue=getattr(lp,field); setattr(lp,field,value); ok('mutated LP '+field,consume(a) is None); setattr(lp,field,oldvalue)
oldlp=w._target_logits_processor; w._target_logits_processor=NS(**vars(oldlp)); ok('replaced LP object rejected',consume(a) is None); w._target_logits_processor=oldlp
cuda.stream_key=202; ok('nonforward consumption stream rejected',consume(a) is None); cuda.stream_key=101
m.forward_stream=NS(cuda_stream=202); ok('replaced forward stream rejected',consume(a) is None); m.forward_stream=NS(cuda_stream=101)
old=r.backend._graphs[Key()]; r.backend._graphs[Key()]=object(); ok('replacement backend graph rejected',consume(a) is None); r.backend._graphs[Key()]=old
for k,v in [('stream_idx',1),('variant_label','other'),('attention_variant','other')]:
    r._replay_graph_key=Key(**{k:v}); ok('reject replay '+k,consume(a) is None)
r._replay_graph_key=Key(); original=parallel.tp_group; parallel.tp_group=NS(ranks=list(range(8)),rank_in_group=0,world_size=8); ok('new group identity rejected',consume(a) is None); parallel.tp_group=original
m.decode_cuda_graph_runner=DecodeCudaGraphRunner(); ok('new runner rejected',consume(a) is None); m.decode_cuda_graph_runner=r
for field in ['require_gathered_buffer','require_mlp_sync','enable_two_batch_overlap','enable_prefill_cp','enable_pdmux','record_nolora_graph','ragged_verify_mode','is_dllm']:
    setattr(r,field,True); ok('reject runner '+field,consume(a) is None); setattr(r,field,False)
r.attention_graph_variants=object(); ok('reject variants',consume(a) is None); r.attention_graph_variants=None
saved=a.state.output; a.on_capture_start(r); ok('recapture clears permission',consume(a) is None); ok('recapture retains output storage',a.state.output is saved)
w,m,a,r=ready(); generation=a.state.generation; a.on_capture_abort(r); ok('abort invalidates all keys',consume(a) is None and a.state.generation==generation+1 and not a.state._runner_refs)
w,m,a,r=ready(); saved=a.state.output; r.max_bs=8; a.on_capture_start(r); ok('growth rejects without freeing output',a.state.output is saved and consume(a) is None)
w,m,a,r=ready(); a.state.output=Tensor(32); ok('mutated output pointer rejected',consume(a) is None)
for obj,field,value in [(parallel,'attn_dp_enabled',True),(parallel,'attn_cp_size',2),(parallel,'attn_dcp_size',2),(parallel,'pp_size',2),(parallel,'tp_size',4),(disagg,'enable_pdmux',True),(execution.overlap,'enable_two_batch_overlap',True)]:
    old=getattr(obj,field); setattr(obj,field,value); ok('worker admission '+field,ns['maybe_install_target_argmax_capture'](w,enabled=True) is None); setattr(obj,field,old)
# Execute exact overlaid capture wrapper with fake _capture: success and failure order.
tree=ast.parse((O/'model_executor/runner/decode_cuda_graph_runner.py').read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DecodeCudaGraphRunner'); method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
space={}; exec(compile(ast.Module(body=[method],type_ignores=[]),'capture-overlay','exec'),space)
for failure in [False,True]:
    events=[]; hook=NS(on_capture_start=lambda r:events.append('start'),on_capture_abort=lambda r:events.append('abort'))
    def inner():
        events.append('body')
        if failure: raise RuntimeError('synthetic backend failure')
    fake=NS(model_context=contextlib.nullcontext,model_runner=NS(capture_tail_hooks=[lambda *x:None,hook]),_capture=inner)
    try: space['capture'](fake)
    except RuntimeError: pass
    ok('actual capture wrapper order '+str(failure),events==(['start','body','abort'] if failure else ['start','body']))
# Assert notification is immediately after the single backend install call structurally.
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='capture_one_shape')
pairs=[]
for node in ast.walk(method):
    for _,value in ast.iter_fields(node):
        if isinstance(value,list):
            for i,n in enumerate(value[:-1]):
                if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='capture_one': pairs.append(isinstance(value[i+1],ast.For))
ok('post install notification ordering',pairs==[True])
report={'scope':'CPU fakes execute real adapter/core and actual extracted capture wrapper; no CUDA, NCCL or serving claim','checks':checks,'count':len(checks),'files':{str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*.py')}}
(H/'TARGET_ARGMAX_OVERLAY_CPU_RESULT.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'count':len(checks),'status':'PASS'}))

