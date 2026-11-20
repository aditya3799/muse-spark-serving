"""Prepare matched cp.async controls for the new SM90 residency result."""
from pathlib import Path
import copy,json
P=Path(__file__).resolve().parent
OLD=P.parents[2]/'swarm_2026-11-16/moe/phase4_hour'
jobs=[]
for source in json.loads((OLD/'producer_compile_jobs.json').read_text())['jobs']:
    if source['configuration']['block_shape'][0]!=8 or source['policy']!='tma_b_only':
        continue
    for stages,ctas in ((3,1),(3,2),(4,1),(4,2)):
        job=copy.deepcopy(source)
        job['configuration']=copy.deepcopy(source['baseline'])
        job['configuration']['num_stages']=stages
        job['configuration']['num_ctas_per_sm']=ctas
        job['policy']='cp_async'
        job['job_id']='%s_cp_async_m8_s%d_cta%d'%(source['stage'],stages,ctas)
        jobs.append(job)
assert len(jobs)==8
(P/'occupancy_control_jobs.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n')
s=(P/'offline_occupancy_sm90.py').read_text().replace("P/'occupancy_jobs.json'", "P/'occupancy_control_jobs.json'")
s=s.replace('default=12','default=8')
(P/'offline_occupancy_controls.py').write_text(s)
print('8 matched cp.async controls prepared')
