"""One-GPU package/lifetime/memory probe; mocked TP peers, never NCCL evidence."""
import gc,importlib.util,json,logging,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
import sys,types,importlib
ROOT=Path('/mnt/d/OpenKernelChallenge/serving')
SOURCE=ROOT/'research_2026-11-19_final/assembled/python/sglang'
for name,p in [('sglang',SOURCE),('sglang.srt',SOURCE/'srt'),('sglang.srt.speculative',SOURCE/'srt/speculative')]:
 m=types.ModuleType(name);m.__path__=[str(p)];sys.modules[name]=m
sys.path.insert(0,str(ROOT/'research_2026-11-19/prefill'))
from real_row_chain import checkpoint
HERE=Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    path=SOURCE/'srt/speculative/dflash_projection_row.py'
    spec=importlib.util.spec_from_file_location('packaged_row',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    (weight,kv,norm,knorm),receipt=checkpoint();del kv,norm,knorm
    class Linear(torch.nn.Linear):
        def __init__(self):
            super().__init__(weight.shape[1],weight.shape[0],bias=False,device='meta',dtype=torch.bfloat16)
            self.weight=torch.nn.Parameter(weight,requires_grad=False);self.reference=None
        def forward(self,x):
            result=torch.nn.functional.linear(x,self.weight)
            self.reference=result
            return result
    linear=Linear()
    class Model:
        fc=linear
        def project_target_hidden(self,x,*,projection=None):return (projection or self.fc)(x)
    class Group:
        world_size=8;rank_in_group=0;cpu_group=None
        def __init__(self):self.template=None;self.calls=0;self.timing_mode=False
        def all_gather_into_tensor(self,output,local):
            self.calls+=1
            if self.timing_mode:
                output.zero_();output[:local.shape[0]].copy_(local);linear.reference=None
                return
            assert linear.reference is not None
            output.copy_(linear.reference);output[:local.shape[0]].copy_(local)
            linear.reference=None
        def all_gather(self,local,dim):
            assert dim==0 and self.template is not None
            self.calls+=1;out=self.template.clone();out[:local.shape[0]].copy_(local)
            linear.reference=None
            return out
    group=Group()
    def mock_gather(result,value,group=None):
        for i in range(len(result)):result[i]=value
    mod.dist.all_gather_object=mock_gather
    mod.dist.all_reduce=lambda *a,**k:None
    os.environ['SGLANG_EXPERIMENTAL_DFLASH_ROW_FC']='1'
    torch.cuda.synchronize();torch.cuda.empty_cache()
    before_alloc=torch.cuda.memory_allocated();before_reserved=torch.cuda.memory_reserved()
    # This standalone process owns its measurement; the production helper must
    # preserve the serving process's global allocator high-water history.
    torch.cuda.reset_peak_memory_stats()
    started=time.time();router=mod.create_prefill_projection(Model(),group,[512,16384],ordinary_tp=True,measure=False)
    peak_alloc=torch.cuda.max_memory_allocated()-before_alloc
    peak_reserved=torch.cuda.max_memory_reserved()-before_reserved
    assert router is not None and router.published=={(512,40960),(16384,40960)}
    report=dict(complete=False,scope='Single GPU, real FC weights and actual packaged native plans. CPU votes and seven peer row outputs mocked. Real distributed controls are separate Gloo reports; no NCCL or H200 claim.',checkpoint=receipt,registration_seconds=time.time()-started,
        published=sorted(router.published),incremental_peak_allocated=peak_alloc,incremental_peak_reserved=peak_reserved,
        fixed_scratch_bytes=sum(x.numel() for x in router.workspaces.values()),startup_extra_allocated_below_decimal_2GB=peak_alloc<2_000_000_000,
        startup_extra_reserved_below_decimal_2GB=peak_reserved<2_000_000_000,cases={})
    (HERE/'row_package_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    assert peak_alloc<2_000_000_000 and peak_reserved<2_000_000_000,report
    gen=torch.Generator(device='cuda').manual_seed(19261120)
    x=torch.randn((512,40960),device='cuda',dtype=torch.bfloat16,generator=gen)
    group.template=torch.nn.functional.linear(x,weight)
    actual=router(x);assert torch.equal(actual.view(torch.int16),group.template.view(torch.int16))
    report['cases']['registered_stream_exact']=True
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual=router(x)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(actual.view(torch.int16),group.template.view(torch.int16))
    report['cases']['unknown_stream_fallback_still_gathers']=True
    (HERE/'row_package_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    # Capture on the registered stream. Peer template storage is updated in place
    # on every replay; no stale-cache or fake speed measurement is inferred.
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):actual=router(x)
    for mutation in ('integer','magnitude'):
        if mutation=='integer':x.random_(-2,3,generator=gen)
        else:x.normal_(generator=gen);x[:,::7].mul_(8)
        group.template.copy_(torch.nn.functional.linear(x,weight))
        graph.replay();torch.cuda.synchronize()
        assert torch.equal(actual.view(torch.int16),group.template.view(torch.int16))
    report['cases']['changed_input_graph_replay_exact']=True
    # The actual timing code executes, but this mock only fills peer rows.
    # Validate record/dispatch-failure mechanics; never record these as NCCL
    # latency or as a reason to enable a production candidate.
    group.timing_mode=True
    timing=mod._row_startup_timing(router,512,40960,gen,rounds=2)
    assert timing['active'] and all(len(v)==2 for v in timing['samples'].values())
    selected=router.plans[((512,40960),mod._stream(x))]
    saved_invoke=selected.invoke;selected.invoke=lambda *args:15
    try:rejected=mod._row_startup_timing(router,512,40960,gen,rounds=2)
    finally:selected.invoke=saved_invoke
    assert not rejected['active']
    report['cases']['transport_timing_records_and_dispatch_veto']=True
    report['timing_probe_scope']='Mocked peers; tests timing and failed-dispatch control flow only. No NCCL timing is reported.'
    try:router.close()
    except RuntimeError:report['cases']['close_rejects_live_graph']=True
    else:raise AssertionError('graph lifetime guard absent')
    del graph,actual;gc.collect();router.close(graphs_destroyed=True)
    report['cases']['close_after_graph_destruction']=True
    assert not router.published and not router.plans
    group.timing_mode=False
    saved_timing=mod._row_startup_timing
    mod._row_startup_timing=lambda *a,**k:dict(active=True,samples={'native_us':[1.]*12,'control_us':[1.]*12,'combined_us':[2.]*12},scope='injected slower transport for rejection test')
    try:declined=mod.create_prefill_projection(Model(),group,[512],ordinary_tp=True,measure=True)
    finally:mod._row_startup_timing=saved_timing
    assert declined is None
    report['cases']['slower_collective_disables_shape']=True
    report['complete']=True
    import hashlib
    report['package_source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'row_package_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checkpoint','scope')}),flush=True)

if __name__=='__main__':
    forward_stream=torch.cuda.Stream()
    with torch.cuda.stream(forward_stream):main()
