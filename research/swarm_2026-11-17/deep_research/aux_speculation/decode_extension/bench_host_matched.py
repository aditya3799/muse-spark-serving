"""Matched CPU-only original/hot wrapper bookkeeping with baseline A/A."""
import contextlib,json,random,runpy,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
P=Path(__file__).resolve().parent;functions={}
for hot in [False,True]:
    label='hot' if hot else 'original'
    sys.argv=[str(P/'test_decode_contract.py'),'--device','cpu','--out',str(P/('matched_'+label+'_contract.json'))]+(['--hot'] if hot else [])
    f=runpy.run_path(str(P/'test_decode_contract.py'));r=f['r'];lease=f['lease'];worker=f['worker'];fb=SimpleNamespace(batch_size=1,forward_mode=f['mode'].TARGET_VERIFY)
    original=f['method'](f['S']/'srt/model_executor/runner/decode_cuda_graph_runner.py','DecodeCudaGraphRunner','execute',f['renv'])
    output=f['Logits'](next_token_logits=r.buffers.next_token_logits_buffer,hidden_states=lease.buffer)
    r.backend=SimpleNamespace(replay_session=contextlib.nullcontext,replay=lambda *args,output=output:output)
    worker._append_target_hidden_to_draft_kv_by_loc_impl=lambda **kwargs:None
    def baseline(r=r,fb=fb,original=original,worker=worker):
        result=original(r,fb);worker._append_target_hidden_to_draft_kv_by_loc_impl(target_hidden=result.hidden_states)
    def body(*args,r=r,fb=fb,worker=worker,**kwargs):
        result=r.execute(fb);worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=result.hidden_states)
    worker._forward_batch_generation_with_aux_output=body
    functions['baseline_'+label]=baseline;functions[label]=lambda worker=worker:worker.forward_batch_generation(None)
samples={name:[] for name in functions};rng=random.Random(185);n=1500
with patch.object(torch.cuda,'current_stream',lambda device=None:f['stream']):
    for _ in range(100):
        for fn in functions.values():fn()
    for _ in range(15):
        names=list(functions);rng.shuffle(names)
        for name in names:
            start=time.perf_counter_ns()
            for _ in range(n):functions[name]()
            samples[name].append((time.perf_counter_ns()-start)/n/1000)
med={name:statistics.median(values) for name,values in samples.items()}
report={'scope':'CPU-only exact source wrappers; fake current_stream and no-op backend/consumer. Actual CUDA API cost and full serving excluded. Matched randomized variants and duplicate original runner baseline.','iterations':n,'samples_us':samples,'median_us':med,'original_added_us':med['original']-med['baseline_original'],'hot_added_us':med['hot']-med['baseline_hot'],'absolute_reduction_us':med['original']-med['hot'],'AA_ratio':med['baseline_original']/med['baseline_hot']}
(P/'HOST_OVERHEAD_MATCHED.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='samples_us'}))
