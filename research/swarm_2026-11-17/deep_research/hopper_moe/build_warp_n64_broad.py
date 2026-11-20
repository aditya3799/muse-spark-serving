from pathlib import Path
import copy,json
P=Path(__file__).resolve().parent;OLD=P.parents[2]/'swarm_2026-11-16/moe/phase4_hour';source=json.loads((OLD/'producer_compile_jobs.json').read_text())['jobs']
jobs=[]
for m,stage,producer,stages in [(32,'w13','tma_b',3),(32,'w13','tma_b',4),(128,'w13','cp_async',4),(256,'w13','tma_b',4),(256,'w2','tma_b',4)]:
 old=next(j for j in source if j['stage']==stage and j['policy']=='tma_b' and any(lo<m*8<=hi for lo,hi in j['intervals']))
 job=copy.deepcopy(old);c=copy.deepcopy(old['baseline'] if producer=='cp_async' else old['configuration']);c['warp_shape'][1]=64;c['num_stages']=stages;c['num_ctas_per_sm']=2
 assert c['block_shape'][1]//64%4==0 and c['warp_shape'][0]*2<=255
 job.update(configuration=c,policy=producer,tokens=m,job_id='%s_m%d_wn64_%s_s%d_cta2'%(stage,m,producer,stages));jobs.append(job)
(P/'warp_n64_broad_jobs.json').write_text(json.dumps({'scope':'Five justified geometry probes, unchanged block and K partition. M32 TMA tests tight128reg CTA2 budget atstage3/4; M128 cpasync isolates geometry without TMA; M256 TMA tests192 accumulator register pressure forbothstages. No ineligibleW2N128 or largeBMmandatoryspill cases.','jobs':jobs},indent=2)+'\n')
s=(P/'offline_warp_n64_sm90.py').read_text().replace("P/'warp_n64_jobs.json'","P/'warp_n64_broad_jobs.json'").replace('default=6','default=5').replace('and k.num_math_threads==256','and k.num_math_threads in (128,256)')
(P/'offline_warp_n64_broad_sm90.py').write_text(s)
print('Prepared5broaderconfigs; notcompiled')
