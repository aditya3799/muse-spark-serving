"""Bounded cross-stream prime ownership gate; no serving/time claim."""
import argparse,hashlib,json
from pathlib import Path
import torch
from phase_recorder import PhaseRecorder
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
out=a.out;out.mkdir(exist_ok=True)
torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>(512+32)*2**20
torch.cuda.reset_peak_memory_stats()
rec=PhaseRecorder(capacity=2,max_phases=2,max_requests=4,every=1,max_records=12)
setup=torch.cuda.Stream();other_setup=torch.cuda.Stream();forward=torch.cuda.Stream()
counts=torch.zeros(4,dtype=torch.int32,device='cuda')
# Initial data setup is ordered into the serving stream explicitly.
forward.wait_stream(torch.cuda.current_stream())
expected=[]
for generation in range(6):
    assert not rec.pending
    with torch.cuda.stream(setup):
        torch.cuda._sleep(1000000)  # Bounded single-block delay; <ms-scale normally.
        rec.prime()
    if generation%2:
        with torch.cuda.stream(other_setup):rec.prime()
    values=[generation+1,3,8,4]
    with torch.cuda.stream(forward):
        counts.copy_(torch.tensor(values,device='cuda',dtype=torch.int32))
        assert rec.begin({'generation':generation})
        assert rec._prime_done is None
        with rec.phase('identity'):counts.add_(0)
        rec.end(counts)
        completed=torch.cuda.Event();completed.record()
    # Synchronization belongs to the test harness, not the recorder.
    completed.synchronize();rec.poll()
    assert rec.records[-1]['commit_lengths']==values
    expected.append(values)
# Profile the first cross-stream consumption after another asynchronous prime.
with torch.cuda.stream(setup):rec.prime()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.cuda.stream(forward):
        with torch.profiler.record_function('cross_stream_observed_step'):
            assert rec.begin({'generation':'profile'})
            with rec.phase('identity'):counts.add_(0)
            rec.end(counts)
prof.export_chrome_trace(str(out/'probe.trace.json'))
trace=json.loads((out/'probe.trace.json').read_text())['traceEvents']
span=next(e for e in trace if e.get('name')=='cross_stream_observed_step')
inside=[e for e in trace if span['ts']<=e.get('ts',-1)<span['ts']+span['dur']]
assert not [e for e in inside if 'synchron' in e.get('name','').lower()]
waits=[e['name'] for e in inside if 'streamwaitevent' in e.get('name','').lower()]
assert waits, 'first sampled stream must wait for cross-stream prime'
torch.cuda.synchronize();rec.poll()
assert rec.records[-1]['commit_lengths']==expected[-1]
assert torch.cuda.max_memory_allocated()<32*2**20
report={'status':'PASS_CROSS_STREAM_PRIME_OWNERSHIP','generations':6,'chained_prime_generations':3,
        'all_histograms_exact':True,'device_wait_seen':waits,'no_host_sync_in_observed_step':True,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'source_sha256':hashlib.sha256(Path(__file__).with_name('phase_recorder.py').read_bytes()).hexdigest(),
        'limitations':['Recorder still requires single-thread serial sampled-step use','No real SGLang/model execution or observer-overhead qualification']}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
