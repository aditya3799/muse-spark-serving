from pathlib import Path
from copy import deepcopy
import json,hashlib
P=Path(__file__).resolve().parent
jobs=[]
for name in ('occupancy_jobs.json','occupancy_control_jobs.json'):
 for old in json.loads((P/name).read_text())['jobs']:
  c=old['configuration']
  if old['stage']!='w13' or c['num_stages']!=4:continue
  # Prior TMA stage4 CTA1 from yesterday is a needed matched geometry arm.
  for ctas in ((1,2) if c['num_ctas_per_sm']==2 else ()):
   job=deepcopy(old);conf=job['configuration'];conf['warp_shape'][1]=64;conf['num_ctas_per_sm']=ctas
   job['job_id']='w13_%s_m8_wn64_s4_cta%d'%(old['policy'],ctas)
   job['matched_warp32_configuration']=old['configuration']
   jobs.append(job)
assert len(jobs)==6 and len({j['job_id'] for j in jobs})==6
out={'status':'prepared_not_compiled','scope':'W13 only. Same block shape, warp M/K, native MMA expression and K partition; warpN32 to64 changes loader/epilogue ownership and requires H200 bitwise gates. W2 N_WARPS2 would violate WGMMA divisibility and is excluded.','jobs':jobs}
(P/'warp_n64_jobs.json').write_text(json.dumps(out,indent=2)+'\n')
s=(P/'offline_occupancy_sm90.py').read_text()
s=s.replace("P/'occupancy_jobs.json'","P/'warp_n64_jobs.json'").replace('default=12','default=6')
s=s.replace('assert k.block_shape==baseline.block_shape and k.warp_shape==baseline.warp_shape',"assert k.block_shape==baseline.block_shape and k.warp_shape[0]==baseline.warp_shape[0] and k.warp_shape[2]==baseline.warp_shape[2]\n  assert k.warp_shape[1]==64 and baseline.warp_shape[1]==32 and k.num_math_threads==256")
(P/'offline_warp_n64_sm90.py').write_text(s)
print('Prepared six W13 warpN64 builds, no compilation')

