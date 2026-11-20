"""Actual pinned accept-buffer alias negative; CPU only."""
from __future__ import annotations
import ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace as N
import torch
import prefix_width_scaffold as loader
P=Path(__file__).parent;ROOT=P.parents[3];SOURCE=ROOT/'sglang/python/sglang/srt/speculative/dflash_worker_v2.py'
tree=ast.parse(SOURCE.read_text());owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DFlashWorkerV2')
ns={'torch':torch};names=('_ensure_accept_bonus_buffers','_next_accept_bonus_buffers')
loader.load([next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in names],ns)
W=type('SourceBuffers',(),{name:ns[name] for name in names})
def worker():
 w=W();w.device='cpu';w.block_size=8;w._accept_bonus_buffer_cap=0;w._accept_bonus_buffer_slot=0;return w
w=worker();first=w._next_accept_bonus_buffers(3);first[0].copy_(torch.tensor([0,4,1],dtype=torch.int32))
saved=first[0].clone();first_pointer=first[0].data_ptr();slot_after_first=w._accept_bonus_buffer_slot
tail=w._next_accept_bonus_buffers(1);tail[0].fill_(3)
assert tail[0].data_ptr()==first_pointer
assert first[0].tolist()==[3,4,1] and saved.tolist()==[0,4,1]
assert first[1].data_ptr()!=tail[1].data_ptr() # two-slot commit differs from single accept
assert slot_after_first==1 and w._accept_bonus_buffer_slot==0
# Proper future design computes both stages in dedicated temporary arrays, then
# reserves the public final-output ring exactly once after original-order merge.
w2=worker();stage1=torch.tensor([0,4,1],dtype=torch.int32);survivors=torch.tensor([1])
private_tail=torch.tensor([3],dtype=torch.int32);merged=stage1.clone();merged[survivors]=4+private_tail
final=w2._next_accept_bonus_buffers(3);final[0].copy_(merged);final[1].copy_(merged+1)
assert final[0].tolist()==[0,7,1] and final[1].tolist()==[1,8,2]
assert stage1.tolist()==[0,4,1] and w2._accept_bonus_buffer_slot==1
r={'status':'CPU_SOURCE_ACCEPT_BUFFER_ALIAS_NEGATIVE_PASS','actual_source_methods':list(names),
 'counterexample':{'first_stage':[0,4,1],'survivor_original_row':1,'compact_tail_accept':[3],'corrupted_first_stage_view':first[0].tolist(),'dedicated_copy':saved.tolist()},
 'required_integration':['Dedicated/copy stage1 continuation decisions before compacttail','Private tail accept workspace; do notreuse shared _accept_len_buf while stage1 decisionslive','Allocate finalfullbatch output/commit/bonus/publication ring ONCE afteroriginalordermerge','Publish finalnewsequence lengths ONCE; firststage isnot a schedulerresult'],
 'not_invalidated':'Prior idealstate andfullFCstaging proofs do notcallstock _accept_block twice; thisis an additional concreteproductionintegrationblock',
 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(P/'SEGMENTED_ACCEPT_BUFFER_CPU.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))