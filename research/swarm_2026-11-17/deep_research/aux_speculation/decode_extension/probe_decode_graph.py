"""Real native decode backend + BCG/dedup, distinct pointers and async leases."""
import argparse,contextlib,json,runpy,sys,time,types
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--fused',action='store_true');pa.add_argument('--hot',action='store_true');pa.add_argument('--hidden',type=int,default=64);a=pa.parse_args()
contract=a.out.with_name(a.out.stem+'_contract.json');oldargv=sys.argv;sys.argv=[str(P/'test_decode_contract.py'),'--device','cuda','--hidden',str(a.hidden),'--out',str(contract)]+(['--fused'] if a.fused else [])+(['--hot'] if a.hot else [])
f=runpy.run_path(str(P/'test_decode_contract.py'));sys.argv=oldargv
S=f['S'];C=f['C'];r=f['r'];lease=f['lease'];worker=f['worker'];model=f['model'];forward=f['forward'];reference=f['old'];Logits=f['Logits'];mode=f['mode'];extract=f['extract'];method=f['method'];load=f['load']
H=f['H'];reference_model=model
if a.fused:
    imports=load('decode_jit_imports',P.parent/'fusion_store/runtime_imports.py');imports.prepare()
    load('sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit',S/'kernels/ops/layernorm/muse_spark_norm_residual_jit.py')
    load('sglang.kernels.ops.layernorm.muse_spark_norm_residual_aux_jit',P.parent/'fusion_store/norm_residual_aux.py')
    opjit=method(C/'kernels/ops/layernorm/muse_spark_norm_residual.py','FusedPostNormResidualPreNorm','forward_jit')
    torch.manual_seed(778)
    weights=[torch.rand(H,device='cuda').to(torch.bfloat16)+.5 for _ in range(2)]
    alpha=torch.full((H,),.75,device='cuda');beta=torch.full((H,),1.25,device='cuda')
    class Boundary:
        _triton_supported=lambda *args:True
        jit_kernel_kwargs=lambda *args:dict(post_eps=1e-6,pre_eps=1e-6,post_weight=weights[0],pre_weight=weights[1],alpha=alpha,beta=beta)
        def __call__(self,*args,**kwargs):return opjit(self,*args,**kwargs)
    old_decoder=method(P.parent/'base_sources/python/sglang/srt/models/muse_spark.py','MuseSpark1xDecoderLayer','forward',{'_use_split_bf16_router':lambda rows:False})
    new_decoder=method(C/'srt/models/muse_spark.py','MuseSpark1xDecoderLayer','forward',{'_use_split_bf16_router':lambda rows:False})
    class Layer:
        def __init__(self,i,modified):
            self.i=i;self._router_dtype=None;self.operator_dtype=torch.bfloat16;self.is_moe_layer=False
            self._fused_attn_boundary=Boundary();self._fused_ffn_boundary=Boundary() if i<8 else None
            self.self_attn_input_norm=lambda x:x.to(torch.bfloat16)
            self.self_attn=lambda pos,x,fb:(x*.5+.125).to(torch.bfloat16)
            self.mlp=lambda x:(x*.75).to(torch.bfloat16)
            self.post_feedforward_layernorm=lambda x:torch.nn.functional.rms_norm(x,(H,),eps=1e-6)
            self.post_feedforward_residual_gate=lambda x,y:alpha*x+beta*y.float()
            self.forward=MethodType(new_decoder if modified else old_decoder,self)
        def __call__(self,*args,**kwargs):return self.forward(*args,**kwargs)
    def make_model(modified):return SimpleNamespace(embed_norm=None,layers=[Layer(i,modified) for i in range(9)],layers_to_capture=[0,2,4,6,8],residual_dtype=torch.float32,operator_dtype=torch.bfloat16,norm=lambda x:x)
    model=make_model(True);reference_model=make_model(False)
checks=[];report={'started':time.time(),'checks':checks,'fused':a.fused,'hidden':H,'scope':'Exact source DecodeInputBuffers/capture_prepare/execute and full backend capture_one/replay, actual BCG/dedup. Fixture residual model, registry, distributed scopes and pre-carve; no full serving runtime.'}
def save():a.out.write_text(json.dumps(report,indent=2)+'\n')
def check(name,ok):checks.append({'name':name,'passed':bool(ok)});save();assert ok,name
fullenv={'graph_pool_capture_scope':contextlib.nullcontext,'graph_pool_replay_scope':contextlib.nullcontext}
fullmethods=extract(S/'srt/model_executor/runner_backend/full_cuda_graph_backend.py',['capture_one','replay','cleanup'],fullenv,'FullCudaGraphBackend')
graphdir=S/'srt/model_executor/runner_backend_utils/breakable_cuda_graph'
load('sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils',graphdir/'cuda_utils.py')
utils=types.ModuleType('sglang.srt.utils');utils.get_device_module=lambda:torch.cuda;utils.is_hip=lambda:False;utils.is_xpu=lambda:False;utils.get_bool_env_var=lambda name:False;sys.modules[utils.__name__]=utils
runtime=types.ModuleType('sglang.srt.runtime_context');runtime.get_exec=lambda:None;sys.modules[runtime.__name__]=runtime
environ=types.ModuleType('sglang.srt.environ');environ.envs=SimpleNamespace();sys.modules[environ.__name__]=environ
bcg=load('decode_actual_bcg',graphdir/'breakable_cuda_graph.py');dedup=load('decode_actual_dedup',S/'srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py')
buckets=[8,3,1];inputs={bs:torch.full((bs*8,H),.25,dtype=torch.bfloat16,device='cuda') for bs in buckets}
check('distinct_captured_input_pointers',len({x.data_ptr() for x in inputs.values()})==3)
static={bs:r.capture_prepare(bs)[0] for bs in buckets};producer=torch.cuda.Stream();consumer=torch.cuda.Stream();host=torch.cuda.current_stream()
producer.wait_stream(host);consumer.wait_stream(host)
for backend_kind in ['full','bcg_dedup']:
    pool=torch.cuda.graph_pool_handle();capturestream=torch.cuda.Stream();capturestream.wait_stream(host)
    if backend_kind=='full':
        backend=SimpleNamespace(_graphs={},_outputs={},_pool=pool,_capture_stream=capturestream,_cuda_graph_runner=r,_device_module=torch.cuda,_tp_group=SimpleNamespace(barrier=lambda:None),_precarve=SimpleNamespace(measure=contextlib.nullcontext,mint=lambda:None),_reuse_output_buffer=False,_output_buffer=None,_memory_saver_adapter=None,replay_session=contextlib.nullcontext)
        for name,fn in fullmethods.items():setattr(backend,name,MethodType(fn,backend))
        def capture_all():
            for bs in buckets:
                def body(bs=bs):
                    hidden,aux=forward(model,None,None,static[bs],inputs[bs])
                    return Logits(next_token_logits=r.buffers.next_token_logits_buffer[:bs*8],hidden_states=aux,last_hidden_states=hidden)
                backend.capture_one(bs,body)
    else:
        registry=dedup.DedupedCudaGraphRegistry();graphs={};outputs={};shared=[None]
        bm=extract(C/'srt/model_executor/runner_backend/breakable_cuda_graph_backend.py',['_alloc_full_buffer','_copy_output_to_buffer','_slice_output'],{'PPProxyTensors':type('PP',(),{})},'BreakableCudaGraphBackend')
        b=SimpleNamespace(_cuda_graph_runner=r)
        for name,fn in bm.items():setattr(b,name,MethodType(fn,b))
        # Boundaries surround fixture layers; every original arithmetic op runs.
        class BcgLayer:
            def __init__(self,layer):self.layer=layer;self.self_attn_input_norm=layer.self_attn_input_norm
            def __call__(self,*args,**kwargs):
                if self.layer.i%3==0:bcg.break_graph()
                return self.layer(*args,**kwargs)
        original_layers=model.layers;model.layers=[BcgLayer(layer) for layer in original_layers]
        def capture_all():
            for bs in buckets:
                with torch.cuda.stream(capturestream):
                    warm=forward(model,None,None,static[bs],inputs[bs])
                    if shared[0] is None:shared[0]=b._alloc_full_buffer(warm,64)
                    graph=bcg.BreakableCUDAGraph(registry)
                    with bcg.BreakableCUDAGraphCapture(graph,pool=pool,stream=capturestream):
                        out=forward(model,None,None,static[bs],inputs[bs]);b._copy_output_to_buffer(out,shared[0],bs*8)
                graphs[bs]=graph;outputs[bs]=b._slice_output(shared[0],bs*8)
            registry.seal()
        def replay(key,fb):
            graphs[key].replay();hidden,aux=outputs[key]
            return Logits(next_token_logits=r.buffers.next_token_logits_buffer[:key*8],hidden_states=aux,last_hidden_states=hidden)
        backend=SimpleNamespace(replay=replay,replay_session=contextlib.nullcontext)
    r.stream=capturestream;r._capture=capture_all;r.capture();host.wait_stream(capturestream)
    if backend_kind=='bcg_dedup':
        report['dedup_stats']=registry.stats();model.layers=original_layers
    r.backend=backend
    def load_batch(fb,pp):r.raw_num_token=fb.batch_size*8;r.bs=fb.bucket;r._replay_graph_key=fb.bucket
    r.load_batch=load_batch
    observed=[];expected=[];payloads=[]
    worker._append_target_hidden_to_draft_kv_by_loc_impl=lambda **kwargs:(torch.cuda._sleep(300000),observed.append(kwargs['target_hidden'].clone()))
    for cycle,(bs,bucket) in enumerate([(1,1),(5,8),(2,3),(3,3),(1,8),(8,8)]):
        rows=bucket*8
        payload=(torch.arange(rows*H,device='cuda').reshape(rows,H).remainder(31)/8+cycle*.25).to(torch.bfloat16);payloads.append(payload)
        ref=reference(reference_model,None,None,static[bucket],payload)[1];expected.append(torch.cat(ref,-1)[:bs*8])
        producer.wait_stream(host)
        with torch.cuda.stream(producer):
            inputs[bucket].copy_(payload)
            out=r.execute(SimpleNamespace(batch_size=bs,bucket=bucket,forward_mode=mode.TARGET_VERIFY))
        with torch.cuda.stream(consumer):worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=out.hidden_states)
    host.wait_stream(producer);host.wait_stream(consumer);torch.cuda.synchronize()
    for cycle,(actual,wanted) in enumerate(zip(observed,expected)):
        check(backend_kind+'_generation%d'%cycle,torch.equal(actual,wanted))
    check(backend_kind+'_lease_idle',lease._state=='idle')
    # Pointer mutation and a pending generation must block recapture/replay.
    token=lease.begin_produce();lease.end_produce(token,lease.view(8))
    try:r.capture()
    except RuntimeError:check(backend_kind+'_pending_blocks_recapture',True)
    else:check(backend_kind+'_pending_blocks_recapture',False)
    with lease.consume(lease.view(8)):pass
    if backend_kind=='full':backend.cleanup()
    else:registry.close()
report.update(status='PASS',check_count=len(checks),finished=time.time());save();print(json.dumps({'status':'PASS','checks':len(checks)}))
