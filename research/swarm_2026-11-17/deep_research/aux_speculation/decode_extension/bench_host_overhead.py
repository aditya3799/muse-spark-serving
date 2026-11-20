"""CPU-only lower-bound estimate of added Python lease bookkeeping."""
import contextlib,json,random,runpy,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
P=Path(__file__).resolve().parent
hot='--hot' in sys.argv
sys.argv=[str(P/'test_decode_contract.py'),'--device','cpu','--out',str(P/('cpu_host_bench_hot_contract.json' if hot else 'cpu_host_bench_contract.json'))]+(['--hot'] if hot else [])
f=runpy.run_path(str(P/'test_decode_contract.py'));r=f['r'];lease=f['lease'];worker=f['worker'];fb=SimpleNamespace(batch_size=1,forward_mode=f['mode'].TARGET_VERIFY)
original=f['method'](f['S']/'srt/model_executor/runner/decode_cuda_graph_runner.py','DecodeCudaGraphRunner','execute',f['renv'])
output=f['Logits'](next_token_logits=r.buffers.next_token_logits_buffer,hidden_states=lease.buffer)
r.backend=SimpleNamespace(replay_session=contextlib.nullcontext,replay=lambda *args:output)
worker._append_target_hidden_to_draft_kv_by_loc_impl=lambda **kwargs:None

def baseline():
    result=original(r,fb)
    worker._append_target_hidden_to_draft_kv_by_loc_impl(target_hidden=result.hidden_states)

def body(*args,**kwargs):
    result=r.execute(fb)
    worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=result.hidden_states)
worker._forward_batch_generation_with_aux_output=body

def candidate():worker.forward_batch_generation(None)
samples={'baseline':[],'candidate':[]};rng=random.Random(184);n=2000
with patch.object(torch.cuda,'current_stream',lambda device=None:f['stream']):
    for _ in range(100):baseline();candidate()
    for _ in range(9):
        order=['baseline','candidate'];rng.shuffle(order)
        for name in order:
            fn=baseline if name=='baseline' else candidate
            start=time.perf_counter_ns()
            for _ in range(n):fn()
            samples[name].append((time.perf_counter_ns()-start)/n/1000)
median={name:statistics.median(values) for name,values in samples.items()}
report={'scope':'CPU-only exact extracted runner and worker wrappers with no-op backend/consumer and fake current_stream; excludes real CUDA API overhead, model, KV projection and full scheduler. Diagnostic lower-bound comparison, not throughput.','iterations_per_sample':n,'samples_us':samples,'median_us':median,'added_python_us':median['candidate']-median['baseline'],'ratio':median['candidate']/median['baseline'],'same_stream_gpu_operations':'None in lease hot path; source inspection, not CUDA timing.'}
(P/('HOST_OVERHEAD_HOT.json' if hot else 'HOST_OVERHEAD.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
