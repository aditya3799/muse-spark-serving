"""Compile only the four newly corrected Hopper light-W2 K partitions."""
from pathlib import Path
import copy,hashlib,json
P=Path(__file__).resolve().parent
source=P.parent/'coordination/SM90_K_PRESERVING_POLICIES.json'
r=json.loads(source.read_text());jobs=[]
for row in r['rows']:
    if not row['k_changed']:continue
    c=copy.deepcopy(row['configuration'])
    original=copy.deepcopy(c); original.update(copy.deepcopy(row['original_light']))
    assert c['block_shape'][2]==row['baseline']['block_shape'][2]
    assert c['warp_shape'][2]==row['baseline']['warp_shape'][2]
    jobs.append({'stage':row['stage'],'policy':'preserve_baseline_k',
       'job_id':'w2_m%d_k%d_wk%d'%(c['block_shape'][0],c['block_shape'][2],c['warp_shape'][2]),
       'configuration':c,'baseline':original,'serving_default_geometry':row['baseline'],
       'intervals':[[row['first_token_rows'],row['last_token_rows']]]})
assert len(jobs)==4
(P/'k_preserving_jobs.json').write_text(json.dumps({'policy_source':str(source),'policy_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'jobs':jobs},indent=2)+'\n')
s=(P/'offline_occupancy_sm90.py').read_text().replace("P/'occupancy_jobs.json'", "P/'k_preserving_jobs.json'").replace('default=12','default=4')
s=s.replace('assert k.block_shape==baseline.block_shape and k.warp_shape==baseline.warp_shape', "assert k.block_shape[:2]==baseline.block_shape[:2] and k.warp_shape==baseline.warp_shape\n  assert k.block_shape[2]==job['serving_default_geometry']['block_shape'][2] and k.warp_shape[2]==job['serving_default_geometry']['warp_shape'][2]")
(P/'offline_k_preserving_sm90.py').write_text(s)
print('4 K-preserving W2 builds prepared')

