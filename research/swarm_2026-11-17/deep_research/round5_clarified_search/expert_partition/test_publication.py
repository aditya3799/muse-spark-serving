from replica_publication import Ledger,Publication,validate
from replica_ownership_oracle import Session,Budget
from dataclasses import replace
import hashlib,json,pathlib
P=pathlib.Path(__file__).parent
s=Session();s.observe([list(range(8))for _ in range(128)]);base=s.baseline();plan=s.choose(Budget(8),max_copies=8)
cap=[max(v for v in row if v is not None)+1 for row in plan.slots]
hashes={e:hashlib.sha256(bytes([e])*40).hexdigest()for e in range(256)}
neg=[]
def reject(name,fn):
 try:fn()
 except (ValueError,RuntimeError):neg.append(name)
 else:raise AssertionError(name)
pub=Publication(base,cap);pub.enter(0);reject('live_graph_prepare',lambda:pub.prepare(plan,hashes));pub.leave(0)
d=pub.prepare(plan,hashes);reject('new_launch_while_pending',lambda:pub.enter(1));reject('partial_publish',pub.publish)
reject('wrong_generation',lambda:pub.mark_ready(0,'wrong',{}));reject('bad_copy_or_scale_bytes',lambda:pub.mark_ready(0,d,{}))
for rank in range(8):pub.mark_ready(rank,d,{e:hashes[e]for e,h in enumerate(plan.holders)if rank in h})
assert pub.publish() is plan
pub.prepare(plan,hashes);pub.abort();assert pub.active is plan
pub.prepare(plan,hashes);pub.begin_copy();pub.abort();reject('abort_after_partial_copy',lambda:pub.enter(0))
bad=list(plan.holders);bad[0]=();reject('zero_holder_modulo',lambda:validate(replace(plan,holders=tuple(bad)),8,256,cap))
reject('capacity_not_reserved',lambda:validate(plan,8,256,[32]*8))
l=Ledger(1,100);l.allocate('old_resident',0,40);l.allocate('new_resident',0,30);l.allocate('copyA',0,20)
reject('overlapping_copy_peak',lambda:l.allocate('copyB',0,20));l.free('copyA');l.allocate('copyB',0,20);assert l.peak==[90]
reject('full_reallocation_peak',lambda:l.allocate('replacement_full_layer',0,40))
report={'status':'PASS_CPU_PUBLICATION_MODEL_ONLY','negatives':neg,'peak_fixture':[90],'conditions':['All ranks actually quiescent and transfers complete before publication are modeled preconditions, not implemented GPU/distributed proof.','Ledger tracks simultaneously-live buffers; old+new capacity and concurrent temporaries are additive.','Copied hash fixture covers logical expert bytes; actual Humming tensor/schema digest inventory remains required.','Global replicated row numbering is required by token-mod ownership; SP/DP/ragged slices are not admitted.']}
(P/'PUBLICATION_CPU_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
