import argparse,importlib.util,json,random,sys,time,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from resident_lookup import ResidentLookup
parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
R=P.parents[3]
free,total=torch.cuda.mem_get_info()
assert free>768*1024**2, 'insufficient conservative headroom'
torch.cuda.set_per_process_memory_fraction(256*1024**2/total)
torch.cuda.reset_peak_memory_stats()
source=R/'sglang/python/sglang/kernels/ops/speculative/dflash.py'
spec=importlib.util.spec_from_file_location('native_dflash_probe',source);native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
checks=[]

def oracle(h,eos=()):
    best=(-1,None)
    for end in range(4,len(h)-6):
        if h[end-4:end]!=h[-4:] or any(t in eos for t in h[end:end+7]):continue
        k=4
        while k<min(16,end,len(h)) and h[end-k-1]==h[-k-1]:k+=1
        score=(k,end)
        if best[1] is None or score>best[0]:best=(score,(h[-1],*h[end:end+7]))
    return best[1]

rng=random.Random(431)
histories=[]
for i in range(8):
    h=[rng.randrange(1000) for _ in range(500+i*37)]
    if i%3!=0:h+=h[80:96]
    histories.append(h)
state=ResidentLookup(histories,capacity=1024)
# Warm, then capture lookup + actual native CUDA acceptance + verified-history append.
state.lookup()
commit=torch.tensor([1,2,3,4,5,6,7,8],device='cuda',dtype=torch.int32)
out=torch.empty((8,8),device='cuda',dtype=torch.int64)
# The native accept helper returns allocated outputs; keep captured buffers alive.
predict=torch.empty((8,8),device='cuda',dtype=torch.int64)
state.lookup();torch.cuda.synchronize()
lookup_graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(lookup_graph):state.lookup()
for generation in range(8):
    lookup_graph.replay();torch.cuda.synchronize()
    tokens=state.tokens.cpu().tolist();hits=state.hit.cpu().tolist()
    for i,h in enumerate(histories):
        expected=oracle(h)
        assert bool(hits[i])==(expected is not None)
        assert tokens[i][0]==h[-1]
        if expected:assert tuple(tokens[i])==expected
    # Target predictions constructed solely to exercise each reject position;
    # this is acceptance-kernel correctness, not real acceptance quality.
    predict.copy_(state.tokens)
    predict[:,:-1].copy_(state.tokens[:,1:]);predict[:,-1].fill_(45678)
    for i,k in enumerate(commit.cpu().tolist()):
        if k<8:predict[i,k-1]=900000+generation*8+i
    accept,bonus=native.compute_dflash_correct_drafts_and_bonus(candidates=state.tokens,target_predict=predict)
    assert torch.equal(accept+1,commit)
    out[:,:-1].copy_(state.tokens[:,1:]);out[:,-1].zero_()
    out.scatter_(1,accept.long()[:,None],bonus[:,None])
    accepted=out.cpu().tolist()
    state.append(out,commit);torch.cuda.synchronize()
    assert state.status.cpu().tolist()==[1]*8
    for i,h in enumerate(histories):h.extend(accepted[i][:i+1])
    assert state.lengths.cpu().tolist()==list(map(len,histories))
    storage=state.history.cpu().tolist()
    for i,h in enumerate(histories):assert storage[i][:len(h)]==h and storage[i][len(h):]==[-777]*(1024-len(h))
    checks.append({'kind':'changed_graph_lookup_native_CUDA_accept_and_append','generation':generation,'rows':8})
# Capture append independently; changed out/counts must be read on replay.
append_graph=torch.cuda.CUDAGraph()
# Capture records the append; it does not execute it.
with torch.cuda.graph(append_graph):state.append(out,commit)
torch.cuda.synchronize()
for generation in range(4):
    out.fill_(20000+generation);commit.fill_(generation+1)
    append_graph.replay();torch.cuda.synchronize()
    for h in histories:h.extend([20000+generation]*(generation+1))
    storage=state.history.cpu().tolist()
    assert all(storage[i][:len(h)]==h for i,h in enumerate(histories))
checks.append({'kind':'changed_append_graph_replays','generations':4})
# Request slot reorder remains isolated; test actual gather mapping.
state.slots.copy_(torch.tensor([7,0,5,2,6,1,4,3],device='cuda',dtype=torch.int32))
lookup_graph.replay();torch.cuda.synchronize()
for i,slot in enumerate([7,0,5,2,6,1,4,3]):assert state.tokens[i,0].item()==histories[slot][-1]
checks.append({'kind':'unique_request_slot_permutation'})
# Invalid append cannot cross row boundary; poisons the request.
small=ResidentLookup([[1,2,3,4,5,6,7,8]],capacity=8)
before=small.history.clone();small.append(torch.ones((1,8),device='cuda',dtype=torch.int64),torch.ones(1,device='cuda',dtype=torch.int32))
small.lookup();torch.cuda.synchronize()
assert torch.equal(before,small.history) and small.enabled.item()==0 and small.hit.item()==0
checks.append({'kind':'overflow_no_write_slot_poison'})
# Stale epochs cannot read or append to a recycled request's history.
epoch_state=ResidentLookup([list(range(1,40))+[1,2,3,4]],capacity=64)
epoch_state.lookup();torch.cuda.synchronize();assert epoch_state.hit.item()==1
epoch_graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(epoch_graph):epoch_state.lookup()
epoch_state.epochs.fill_(1) # slot recycled; old captured binding still expects0
before=epoch_state.history.clone();beforelen=epoch_state.lengths.clone()
epoch_graph.replay();epoch_state.append(torch.ones((1,8),device='cuda',dtype=torch.int64),torch.ones(1,device='cuda',dtype=torch.int32))
torch.cuda.synchronize()
assert epoch_state.hit.item()==0 and epoch_state.status.item()==0
assert torch.equal(before,epoch_state.history) and torch.equal(beforelen,epoch_state.lengths) and epoch_state.enabled.item()==1
epoch_state.bind_requests([0],[1]);epoch_graph.replay();torch.cuda.synchronize();assert epoch_state.hit.item()==1
# Even corrupted out-of-range slot IDs are masked in kernels, not dereferenced.
for slot in (-1,1):
    epoch_state.slots.fill_(slot);epoch_state.lookup()
    epoch_state.append(torch.ones((1,8),device='cuda',dtype=torch.int64),torch.ones(1,device='cuda',dtype=torch.int32))
    torch.cuda.synchronize();assert epoch_state.hit.item()==0 and epoch_state.status.item()==0
    assert torch.equal(before,epoch_state.history)
try: state.bind_requests([0]*8,[0]*8)
except ValueError: pass
else: raise AssertionError('duplicate request slots accepted')
checks.append({'kind':'stale_epoch_no_read_no_write_new_owner_not_poisoned_invalid_slot_masked_duplicate_bind_rejected'})

# Stop token invalidates only that source continuation.
eos=ResidentLookup([[1,2,3,4,5,99,7,8,9,10,11,12,1,2,3,4]],capacity=32,eos_ids=(99,))
eos.lookup();torch.cuda.synchronize();assert eos.hit.item()==0
checks.append({'kind':'EOS_source_rejection'})
# Real-size input component; changed requests, no model-performance inference.
longhist=[]
for i in range(4):
    h=[rng.randrange(202048) for _ in range(74160)];h+=h[100:116];longhist.append(h)
longstate=ResidentLookup(longhist,capacity=75008)
longstate.lookup();torch.cuda.synchronize()
assert longstate.hit.cpu().tolist()==[1]*4
assert all(tuple(row)==oracle(h) for row,h in zip(longstate.tokens.cpu().tolist(),longhist))
# CUDA-event component timing only, no CPU dispatch/sync, fixed synthetic data.
a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
a.record()
for _ in range(40):longstate.lookup()
b.record();b.synchronize()
lookup_us=a.elapsed_time(b)*1000/40
checks.append({'kind':'long_context_exact_matches','rows':4,'context_length':74176})
report={'status':'PASS_LOCAL_GPU_COMPONENT_ONLY','checks':checks,'device':torch.cuda.get_device_name(),'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved(),'synthetic_long_context_lookup_us':lookup_us,'resident_module_sha256':hashlib.sha256((P/'resident_lookup.py').read_bytes()).hexdigest(),'native_accept_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'limits':['No Muse forward, real acceptance quality, target/draft KV, TP8 collective or serving speedup.','History lifecycle/unique slot admission and actual worker integration remain held.','One GPU component timing, not H200/full-step timing; CPU readback to skip draft is excluded.','Resident history is confirmed token IDs, distinct from attention KV cache.']}
assert report['peak_torch_allocated_bytes']<256*1024**2
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('checks','limits')}))
