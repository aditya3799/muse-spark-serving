from pathlib import Path
import json,copy,sys,hashlib,ast
from types import SimpleNamespace
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from residency_view import residency_layer_view
jobs=json.loads((P.parents[2]/'swarm_2026-11-16/moe/phase4_hour/producer_compile_jobs.json').read_text())['jobs']
configs={};metas={};seen=set()
class Meta:
 def __init__(self,d):self.d=d
 def to_str(self):return json.dumps(self.d)
for job in jobs:
 stage=job['stage'];b=job['baseline'];key=(stage,tuple(tuple(x) for x in job['intervals']))
 if key in seen:continue
 seen.add(key);metas[stage]=Meta(b)
 for lo,hi in job['intervals']:configs.setdefault(stage,[]).append([lo,hi,copy.deepcopy(b)])
for v in configs.values():v.sort(key=lambda x:x[0])
layer=SimpleNamespace(fp32_configs=configs,humming_metas=metas,_parameters={},_buffers={})
original=copy.deepcopy(configs);rows=[]
for producer in ('cp_async','tma_b_only','tma_b'):
 for stages in (3,4):
  view,m=residency_layer_view(layer,stages,2,90,producer)
  assert layer.fp32_configs==original and view is not layer
  assert view.humming_metas is layer.humming_metas and view._parameters is layer._parameters
  for stage,intervals in configs.items():
   for i,(lo,hi,c) in enumerate(intervals):
    d=view.fp32_configs[stage][i][2]
    assert d['block_shape']==c['block_shape'] and d['warp_shape']==c['warp_shape']
    if c['block_shape'][0]!=8:assert c==d
  rows.append({'producer':producer,'stages':stages,'changed_intervals':[{'stage':r['stage'],'lo':r['lo'],'hi':r['hi']} for r in m['changed']]})
invalid=0
for sm,st,cta,prod in [(80,3,2,'cp_async'),(90,2,2,'cp_async'),(90,3,3,'cp_async'),(86,3,2,'tma_b'),(90,3,2,'unknown')]:
 try:residency_layer_view(layer,st,cta,sm,prod)
 except ValueError:invalid+=1
 else:raise AssertionError('guard not fired')
for f in ('residency_probe_sm90.py','residency_probe_sm86.py'):
 ast.parse((P/f).read_text(encoding='utf-8-sig'))
r={'status':'cpu_contract_passed','source_sha256':hashlib.sha256((P/'residency_view.py').read_bytes()).hexdigest(),'original_interval_count':{k:len(v) for k,v in configs.items()},'invalid_policy_guards':invalid,'rows':rows,'scope':'Metadata and source parse only; no runtime or compilation'}
(P/'residency_cpu_contract.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
