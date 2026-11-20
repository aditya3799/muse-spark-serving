import argparse,hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from resident_frontier_lookup import ResidentLookup
pa=argparse.ArgumentParser();pa.add_argument('--out',required=True);args=pa.parse_args();O=Path(args.out);O.mkdir(parents=True,exist_ok=True)
free,total=torch.cuda.mem_get_info();assert free>768*1024**2
torch.cuda.set_per_process_memory_fraction(128*1024**2/total)
torch.cuda.reset_peak_memory_stats()
histories=[[i%23 for i in range(69)]+[0,1,2,3,4],[i%29 for i in range(87)]+[0,1,2,3,4]]
out=torch.zeros((2,8),device='cuda',dtype=torch.int64);commit=torch.tensor([1,4],device='cuda',dtype=torch.int32)
warm=ResidentLookup(histories,capacity=256);warm.lookup();warm.append(out,commit);torch.cuda.synchronize()
s=ResidentLookup(histories,capacity=256);s.lookup();torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    s.lookup();s.append(out,commit)
checks=[]
for step in range(12):
    lengths=[len(h) for h in histories];anchors=[h[-1] for h in histories]
    counts=[1+step%8,1+(step+3)%8]
    values=[[(anchors[i]+j+1)%(23 if i==0 else 29) for j in range(8)] for i in range(2)]
    out.copy_(torch.tensor(values,device='cuda',dtype=torch.int64));commit.copy_(torch.tensor(counts,device='cuda',dtype=torch.int32))
    s.bind_frontier(torch.tensor([n-1 for n in lengths],device='cuda',dtype=torch.int64),torch.tensor(anchors,device='cuda',dtype=torch.int64))
    g.replay();torch.cuda.synchronize()
    assert s.status.cpu().tolist()==[1,1] and s.hit.cpu().tolist()==[1,1]
    for i,h in enumerate(histories):h.extend(values[i][:counts[i]])
    storage=s.history.cpu().tolist()
    assert all(storage[i][:len(h)]==h for i,h in enumerate(histories))
    before=s.history.clone();length_before=s.lengths.clone()
    g.replay();torch.cuda.synchronize() # same expected pre-frontier => duplicate must not append
    assert s.status.cpu().tolist()==[0,0] and s.hit.cpu().tolist()==[0,0]
    assert torch.equal(before,s.history) and torch.equal(length_before,s.lengths) and s.enabled.cpu().tolist()==[1,1]
    checks.append({'kind':'valid_commit_then_duplicate_graph_no_write','step':step,'counts':counts})
# Same length but wrong anchor must reject, without disabling current request.
correct_prefix=torch.tensor([len(h)-1 for h in histories],device='cuda',dtype=torch.int64)
s.bind_frontier(correct_prefix,torch.tensor([987654,987655],device='cuda',dtype=torch.int64))
before=s.history.clone();g.replay();torch.cuda.synchronize()
assert s.status.cpu().tolist()==[0,0] and s.hit.cpu().tolist()==[0,0] and torch.equal(before,s.history) and s.enabled.cpu().tolist()==[1,1]
checks.append({'kind':'same_length_wrong_anchor_no_write_no_disable'})
# Actual permutation binds matching pre-frontier in the same request-row order.
s.bind_requests([1,0],[0,0]);s.bind_frontier(correct_prefix.flip(0),torch.tensor([histories[1][-1],histories[0][-1]],device='cuda',dtype=torch.int64))
s.lookup();torch.cuda.synchronize();assert s.tokens[:,0].cpu().tolist()==[histories[1][-1],histories[0][-1]]
checks.append({'kind':'permuted_slot_and_frontier_identity'})
report={'status':'PASS_GPU_FRONTIER_DELTA_ONLY','checks':checks,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'source_sha256':hashlib.sha256((P/'resident_frontier_lookup.py').read_bytes()).hexdigest(),'limits':['Only new GPU frontier/duplicate-append guards; target outputs in this fixture are prescribed CPU tokens, not model predictions.','V2 receipt separately tests native CUDA target acceptance.','No whole-worker scheduler wiring, real request lifecycle, TP8 or serving gain.','Caller must bind the pre-verify prefix and anchor, not next-step frontier, and preserve request order and CUDA stream dependencies.']}
assert report['peak_allocated_bytes']<128*1024**2
(O/'results.json').write_text(json.dumps(report,indent=2)+'\n')
(O/'resident_frontier_lookup_snapshot.py').write_bytes((P/'resident_frontier_lookup.py').read_bytes())
print(json.dumps({k:v for k,v in report.items() if k not in ('checks','limits')}))
