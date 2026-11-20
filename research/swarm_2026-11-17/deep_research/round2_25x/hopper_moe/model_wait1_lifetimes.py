"""Worst-case WGMMA lifetime model of pinned source producer schedule."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent

def simulate(stages,tiles,warp_iters=4,wait_n=1,drain=True,early_copy=False,one_bank=False,alias_qb=False):
 pending=[];events=[];banks=1 if one_bank else 2
 def write_bank(bank,kind):
  assert not any(g['bank']==bank for g in pending),(kind,'register overwrite',bank,list(pending))
 def write_stage(stage):
  assert not any(g['stage']==stage for g in pending),('shared_A_overwrite',stage,list(pending))
 def wait(n):
  while len(pending)>n:pending.pop(0)
 for tile in range(tiles):
  stage=tile%stages;remaining=tiles-tile
  for i in range(warp_iters):
   bank=i%banks;nextbank=(i+1)%banks
   if alias_qb:write_bank(nextbank,'prefetch_qb_alias')
   if early_copy and i==0 and remaining>=stages:write_stage((stage+stages-1)%stages)
   pending.append({'tile':tile,'stage':stage,'iter':i,'bank':bank});events.append((tile,i));wait(wait_n)
   if i==warp_iters-2 and stages>2 and remaining>=stages:write_stage((stage+stages-1)%stages)
   write_bank(nextbank,'transform_b')
  if stages==2 and remaining>stages:write_stage(stage)
 if drain:wait(0)
 assert not pending,('epilogue_accumulator_or_shared_read',list(pending))
 assert events==[(tile,i) for tile in range(tiles) for i in range(warp_iters)]
 return len(events)
positive=[{'stages':s,'K_tiles':t,'groups':simulate(s,t)} for s in (3,4) for t in (1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65)]
negative=[]
for name,kw in [('missing_final_drain',{'stages':3,'tiles':1,'drain':False}),('stage2_same_stage_refill',{'stages':2,'tiles':4}),('early_stage_wrap_copy',{'stages':3,'tiles':8,'early_copy':True}),('one_input_bank',{'stages':3,'tiles':2,'one_bank':True}),('wait2_with_two_banks',{'stages':3,'tiles':2,'wait_n':2}),('packed_prefetch_aliases_live_input',{'stages':3,'tiles':2,'alias_qb':True})]:
 try:simulate(**kw)
 except AssertionError as e:negative.append({'mutation':name,'caught':True,'reason':str(e)})
 else:raise AssertionError('Negative not detected: '+name)
# Cross-warpgroup lag: a fast warpgroup may reach next stage while a slow one
# still has its previous-stage final group pending. The canonical CTA barrier
# before producer issue forces all warpgroups through current-stage wait1.
def cross_warpgroup_overwrite(barrier):
 pending_by_wg=[{'stage':1,'iter':2},{'stage':0,'iter':3}]
 if barrier:
  # To reach the same CTA barrier, slow WG runs stage1 iter0..2; wait1 drains stage0.
  pending_by_wg[1]={'stage':1,'iter':2}
 destination=0
 assert all(g['stage']!=destination for g in pending_by_wg),('lagging_warpgroup_shared_A_overwrite',pending_by_wg)
cross_warpgroup_overwrite(True)
try:cross_warpgroup_overwrite(False)
except AssertionError as error:negative.append({'mutation':'remove_CTA_producer_barrier','caught':True,'reason':str(error)})
else:raise AssertionError('Missing CTA barrier negative did not fail')
r={'status':'SOURCE_LIFETIME_MODEL_PASS_NOT_DEVICE_PROOF','positive':positive,'negative':negative,'scope':'A16 NVFP4 GS16 cp.async non-WS stages3/4 WK64; original physical K partitions and native MMA order. Packed QB separate from dequant input registers.','ordering':'Each commit+wait1 leaves at most current group pending; prior group bank is available for next transform. At penultimate iteration producer writes previous shared stage, whose last group was drained at current-stage iter0. Explicit final wait0 before channel/epilogue.','cross_warpgroup_contract':'All warpgroups follow same stage/warpiter loop and execute canonical __syncthreads before producer issue. Barrier joins previous-stage completion proofs across warpgroups. Removing it fails lagging-warpgroup negative.', 'tail_scope':'Integer K tiles and uneven pipeline-ring tails tested; nonmultiple BK element tails rejected by renderer.','limits':['Source-order model; compiler register allocation/instruction scheduling needs SASS audit.','No H200 numerics, racecheck, latency or overlap proof.'],'source_urls':['https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group','https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-09-15-novita-chord-w4a16-moe.md']}
(P/'WAIT1_LIFETIME_MODEL.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({'positive':len(positive),'negative':len(negative),'status':r['status']}))
