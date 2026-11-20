from pathlib import Path
import json,copy,ast
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round3_structural\hopper_moe');R2=P.parents[1]/'round2_25x/hopper_moe';old=json.loads((R2/'LEGAL_M32_JOBS.json').read_text())['jobs'];jobs=[]
for policy in ('light','heavy'):
 base=copy.deepcopy(next(j['configuration'] for j in old if j['label']==policy))
 for kind in ('indexed','grouped'):
  c=copy.deepcopy(base);c['gemm_type']='indexed' if kind=='indexed' else 'grouped_contiguous';jobs.append({'label':kind+'_'+policy,'enabled':kind=='grouped','configuration':c})
c=copy.deepcopy(jobs[-1]['configuration']);jobs.append({'label':'grouped_unpatched_negative','enabled':False,'configuration':c})
(P/'SM86_COMPACT_JOBS.json').write_text(json.dumps({'scope':'Actual M32 scored W13 light/heavy native geometry; only routing layout/scheduler endpoint differs','jobs':jobs},indent=2)+'\n',encoding='utf-8');(P/'launch_probe.py').write_bytes((R2/'launch_probe.py').read_bytes())
