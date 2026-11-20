from pathlib import Path
import json,copy,hashlib
P=Path(__file__).resolve().parent;A=P.parent/'expert_partition/SMALL_M_K_POLICY_AUDIT.json';D=json.loads(A.read_text());OLD=P.parents[2]/'swarm_2026-11-16/moe/phase4_hour';meta=json.loads((OLD/'producer_compile_jobs.json').read_text())['jobs'];unique={}
for row in D['all_1_through_512_contiguous_intervals']:
 if row['arch']!='sm90':continue
 for stage,detail in row.get('stages',{}).items():
  if detail['K_equal']:continue
  configuration=copy.deepcopy(next(j['baseline'] for j in meta if j['stage']==stage));baseline=copy.deepcopy(configuration);baseline.update(detail['light']);configuration.update(detail['proposed_fixed_light'])
  heavy=detail['heavy'];assert configuration['block_shape'][2]==heavy['block_shape'][2] and configuration['warp_shape'][2]==heavy['warp_shape'][2]
  key=stage+json.dumps(configuration,sort_keys=True)
  if key not in unique:
   bm,bn,bk=configuration['block_shape'];unique[key]={'stage':stage,'policy':'small_light_K_repair','job_id':'%s_bm%d_bn%d_bk%d'%(stage,bm,bn,bk),'configuration':configuration,'baseline':baseline,'serving_heavy_K':[heavy['block_shape'][2],heavy['warp_shape'][2]],'intervals':[]}
  unique[key]['intervals'].append([row['first_token_rows'],row['last_token_rows']])
jobs=list(unique.values());assert len(jobs)==9
(P/'small_k_repair_jobs.json').write_text(json.dumps({'scope':'Orthogonal small-M light policy K repair. This deliberately changes scored light partition arithmetic to match heavy K geometry; unchanged-baseline bitwise claim is forbidden. Compile only.','expert_audit_sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'jobs':jobs},indent=2)+'\n')
s=(P/'offline_occupancy_sm90.py').read_text().replace("P/'occupancy_jobs.json'","P/'small_k_repair_jobs.json'").replace('default=12','default=9')
s=s.replace('assert k.block_shape==baseline.block_shape and k.warp_shape==baseline.warp_shape',"assert k.block_shape[:2]==baseline.block_shape[:2] and k.warp_shape[:2]==baseline.warp_shape[:2]\n  assert [k.block_shape[2],k.warp_shape[2]]==job['serving_heavy_K']")
(P/'offline_small_k_repair_sm90.py').write_text(s)
print('Prepared9 unique K-repair configurations, changes scored light arithmetic, not compiled')
