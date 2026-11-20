from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;other=P.parent/'expert_partition/SM90_SMALL_K_COMPILE_JOBS.json';ours=P/'small_k_repair_jobs.json'
a=json.loads(other.read_text())['jobs'];b=json.loads(ours.read_text())['jobs'];matches=[]
for j in a:
 k=next(x for x in b if x['stage']==j['stage'] and x['configuration']['block_shape']==j['configuration']['block_shape'] and x['configuration']['warp_shape']==j['configuration']['warp_shape'])
 differences={key:[value,k['configuration'].get(key)] for key,value in j['configuration'].items() if k['configuration'].get(key)!=value}
 assert not differences,differences
 matches.append({'expert_job':j['job_id'],'our_job':k['job_id'],'matching_explicit_configuration_fields':len(j['configuration'])})
r={'status':'all_four_expert_configs_covered','matches':matches,'expert_manifest_sha256':hashlib.sha256(other.read_bytes()).hexdigest(),'our_manifest_sha256':hashlib.sha256(ours.read_bytes()).hexdigest(),'scope':'Source configuration cross-check only; no numerical equivalence or compilation claim.'}
(P/'small_k_manifest_crosscheck.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
