import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from phase_recorder import PhaseRecorder, install_dflash_recorder

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
free,total=torch.cuda.mem_get_info()
if free < 640*2**20:
    raise RuntimeError('Insufficient memory reserve; TTS must remain untouched')
torch.manual_seed(4421)
x=torch.randn((32,256),device='cuda');w=torch.randn((256,256),device='cuda')
y=torch.empty_like(x);commits=torch.tensor([1,3,8,4],device='cuda',dtype=torch.int32)
# Capture a real graph, warm BLAS before capture. No model downloads.
for _ in range(5):torch.mm(x,w,out=y)
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    torch.mm(x,w,out=y)

class Runner:
    def forward(self,*args,**kwargs):graph.replay();return y
class Target:
    def forward_batch_generation(self,*args,**kwargs):graph.replay();return y
class Draft:
    def project_target_hidden(self,*args,**kwargs):torch.add(y,1,out=y);return y
class Logits:
    def deferred_argmax(self,*args,**kwargs):return torch.argmax(y,dim=-1)
    def finish_deferred_gather(self,*args,**kwargs):return y.clone()
class Worker:
    def __init__(self):
        self.draft_model_runner=Runner();self.target_worker=Target()
        self.draft_model=Draft();self._target_logits_processor=Logits()
    def _wait_for_fused_hicache_load(self,*args,**kwargs):return None
    def _accept_block(self,*args,**kwargs):return commits
    def _append_target_hidden_fused(self,*args,**kwargs):torch.mul(y,2,out=y)
    def _append_target_hidden_sequential(self,*args,**kwargs):torch.mul(y,2,out=y)
    def _append_target_hidden_to_draft_kv_by_loc(self,*args,**kwargs):
        self.draft_model.project_target_hidden()
        self._append_target_hidden_fused()
    def forward_batch_generation(self,batch,*args,**kwargs):
        self._wait_for_fused_hicache_load()
        self.draft_model_runner.forward()
        self.target_worker.forward_batch_generation()
        self._target_logits_processor.deferred_argmax()
        lens=self._accept_block()
        self._append_target_hidden_to_draft_kv_by_loc()
        if getattr(batch,'fail',False):raise ValueError('injected forward failure')
        return SimpleNamespace(accept_lens=lens,can_run_cuda_graph=True,output=y)

batch=SimpleNamespace(seq_lens=[1,2,3,4],forward_mode='TARGET_VERIFY',
    extend_lens=None,is_extend_in_batch=False,can_run_decode_cuda_graph=True)
worker=Worker();baseline=worker.forward_batch_generation(batch).output.clone()
rec=install_dflash_recorder(worker,PhaseRecorder(capacity=4,every=1,max_records=40,max_requests=4))
for i in range(12):
    result=worker.forward_batch_generation(batch)
    # Numerics must be unchanged. Test synchronization is outside recorder.
    assert torch.equal(result.output,baseline)
    commits.copy_(torch.tensor([1+i%8,3,8,4],device='cuda',dtype=torch.int32))
# Drain only here; recorder itself must never synchronize.
torch.cuda.synchronize();rec.poll()
assert len(rec.records)==12
assert rec.records[0]['commit_lengths']==[1,3,8,4]
for i,record in enumerate(rec.records[1:],1):
    assert record['commit_lengths']==[1+(i-1)%8,3,8,4],record
assert all(r['metadata']['target_graph_used'] for r in rec.records)
assert all(len(r['phases'])==8 for r in rec.records)
# Allocation and synchronization audit: profile only a warm wrapped step.
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                      torch.profiler.ProfilerActivity.CUDA]) as prof:
    worker.forward_batch_generation(batch)
    worker.forward_batch_generation(batch)
# Profiler's own trailing synchronization is not recorder work. Inspect in report.
prof.export_chrome_trace(str(out/'probe.trace.json'))
torch.cuda.synchronize();rec.poll()
# Failure path must release the active slot and preserve the original exception.
batch.fail=True
try:worker.forward_batch_generation(batch)
except ValueError as e:assert str(e)=='injected forward failure'
else:raise AssertionError('expected injected failure')
batch.fail=False
torch.cuda.synchronize();rec.poll();assert rec.records[-1]['status']=='forward_error'
rec.uninstall();assert 'forward_batch_generation' not in vars(worker)
assert torch.equal(worker.forward_batch_generation(batch).output,baseline)
# Capture admission check: recorder must not poll or record during capture.
skip=PhaseRecorder(capacity=1,max_phases=2,every=1,max_records=3)
graph2=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph2):
    assert not skip.begin({'capture':True})
    torch.add(x,1,out=y)
assert not skip.pending and not skip.records
# A lazy nested capture may call wrapped helpers. No recorder event may be
# captured; replaying the graph after ring reuse must not overwrite its timers.
nested=install_dflash_recorder(worker,PhaseRecorder(capacity=1,every=1,max_records=3))
assert nested.begin({'test':'nested_capture'})
with nested.phase('parent'):
    nested_graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(nested_graph):
        worker.draft_model.project_target_hidden()
nested.end();torch.cuda.synchronize();nested.poll()
assert nested.records[0]['status']=='nested_capture_invalidates_timing'
assert [p['name'] for p in nested.records[0]['phases']]==['parent']
assert nested.begin({'test':'reuse_after_capture'})
with nested.phase('normal'):worker.draft_model.project_target_hidden()
nested.end();torch.cuda.synchronize();nested.poll()
saved=json.dumps(nested.records)
nested_graph.replay();torch.cuda.synchronize();nested.poll()
assert json.dumps(nested.records)==saved
nested.uninstall()
# Missing optional object method must reject installation before any mutation.
broken=Worker();broken._target_logits_processor.finish_deferred_gather=None
reject=PhaseRecorder(capacity=1,every=1,max_records=1)
try:install_dflash_recorder(broken,reject)
except TypeError:pass
else:raise AssertionError('missing callable must reject install')
assert not reject._undo and 'forward' not in vars(broken.draft_model_runner)
# Slot overflow/drop and nesting errors are explicit, bounded, and recoverable.
tiny=PhaseRecorder(capacity=1,max_phases=1,every=1,max_records=2)
assert tiny.begin({'test':'overflow'})
with tiny.phase('outer'):
    with tiny.phase('omitted'):
        torch.add(x,2,out=y)
tiny.end();torch.cuda.synchronize();tiny.poll()
assert tiny.phase_overflow==1 and len(tiny.records[0]['phases'])==1
# Bench wrapper overhead on same real CUDA graph pipeline, order balanced.
def bench(wrapped,count=150):
    r=None
    if wrapped:r=install_dflash_recorder(worker,PhaseRecorder(every=32,max_records=20,max_requests=4))
    for _ in range(10):worker.forward_batch_generation(batch)
    torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(count):worker.forward_batch_generation(batch)
    torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1e6/count
    if r:r.poll();r.uninstall()
    return elapsed
rows=[]
for order in [(False,True),(True,False),(False,True),(True,False)]:
    rows.append([{'wrapped':v,'wall_us_per_step':bench(v)} for v in order])
report={
 'status':'LOCAL_REAL_CUDA_GRAPH_FIXTURE_ONLY_NOT_FULL_MODEL',
 'device':torch.cuda.get_device_name(),'torch':torch.__version__,
 'records':rec.records,'phase_overflow_checked':True,'capture_skipped':True,
 'nested_capture_safe':True,'partial_install_rejected':True,'exception_preserved':True,'uninstall_restored_instance':True,
 'timing_orders':rows,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
 'source_sha256':hashlib.sha256(Path(__file__).with_name('phase_recorder.py').read_bytes()).hexdigest(),
 'limitations':['Toy worker objects mirror call interfaces but do not execute SGLang/model',
 'CUDA event intervals include stream idle gaps and potentially overlapping work',
 'Profiling adds overhead; separate raw baseline and diagnostic runs are required',
 'No NVLink/NCCL, H200, real requests, scheduler or multi-rank validation']}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('records','timing_orders')}))
