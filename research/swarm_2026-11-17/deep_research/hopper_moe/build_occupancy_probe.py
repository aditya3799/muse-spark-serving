"""Prepare new occupancy experiments, not a replay of prior producer builds."""
from pathlib import Path
import copy, hashlib, json
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
OLD=ROOT/'swarm_2026-11-16/moe/phase4_hour'
jobs=[]
for source in json.loads((OLD/'producer_compile_jobs.json').read_text())['jobs']:
    if source['configuration']['block_shape'][0] != 8:
        continue
    for stages,ctas in ((3,1),(3,2),(4,2)):
        conf=copy.deepcopy(source['configuration'])
        conf['num_stages']=stages
        conf['num_ctas_per_sm']=ctas
        job=dict(source)
        job['configuration']=conf
        job['reference_producer']=source['configuration']
        job['job_id']='%s_%s_m8_s%d_cta%d'%(source['stage'],source['policy'],stages,ctas)
        jobs.append(job)
assert len(jobs)==12
(P/'occupancy_jobs.json').write_text(json.dumps({'scope':'Same block/warp/K and MMA; pipeline-depth/residency changes require runtime gates','jobs':jobs},indent=2)+'\n')
s=(OLD/'offline_all_producer_sm90.py').read_text()
s=s.replace("P=Path(__file__).resolve().parent;V=P.parent/'humming_vendor'", "P=Path(__file__).resolve().parent;OLD=P.parents[2]/'swarm_2026-11-16/moe/phase4_hour';V=OLD.parent/'humming_vendor';sys.path.insert(0,str(OLD))")
s=s.replace("P/'producer_tuning.py'", "OLD/'producer_tuning.py'")
s=s.replace("P/'producer_compile_jobs.json'", "P/'occupancy_jobs.json'")
s=s.replace("default=22", "default=12")
s=s.replace("and k.num_stages==baseline.num_stages", "")
s=s.replace("assert k.block_shape==baseline.block_shape and k.warp_shape==baseline.warp_shape ", "assert k.block_shape==baseline.block_shape and k.warp_shape==baseline.warp_shape")
(P/'offline_occupancy_sm90.py').write_text(s)
print('jobs',len(jobs),'offline source',hashlib.sha256(s.encode()).hexdigest())
