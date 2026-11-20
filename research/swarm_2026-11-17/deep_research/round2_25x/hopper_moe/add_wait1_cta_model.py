from pathlib import Path
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');f=P/'model_wait1_lifetimes.py';s=f.read_text(encoding='utf-8-sig')
needle="r={'status':'SOURCE_LIFETIME_MODEL_PASS_NOT_DEVICE_PROOF'"
insert='''# Cross-warpgroup lag: a fast warpgroup may reach next stage while a slow one
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
'''
assert needle in s;s=s.replace(needle,insert+needle);s=s.replace("'tail_scope':", "'cross_warpgroup_contract':'All warpgroups follow same stage/warpiter loop and execute canonical __syncthreads before producer issue. Barrier joins previous-stage completion proofs across warpgroups. Removing it fails lagging-warpgroup negative.', 'tail_scope':")
f.write_text(s,encoding='utf-8')
