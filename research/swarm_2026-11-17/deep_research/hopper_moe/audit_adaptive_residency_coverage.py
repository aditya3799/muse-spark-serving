from pathlib import Path
import ast,json,copy,sys,hashlib
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P.parents[2];OLD=ROOT/'swarm_2026-11-16';sys.path.insert(0,str(P));from residency_view import residency_layer_view
source=OLD/'coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
tree=ast.parse(source.read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('select_ep_config','choose_partition_configs')];ns={'deepcopy':copy.deepcopy};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
class Meta:
 def __init__(self,d):self.d=d
 def to_str(self):return json.dumps(self.d)
jobs=json.loads((OLD/'moe/phase4_hour/producer_compile_jobs.json').read_text())['jobs'];configs={};metas={};seen=set()
for j in jobs:
 key=(j['stage'],json.dumps(j['intervals']))
 if key in seen:continue
 seen.add(key);metas[j['stage']]=Meta(j['baseline'])
 for lo,hi in j['intervals']:configs.setdefault(j['stage'],[]).append([lo,hi,copy.deepcopy(j['baseline'])])
for v in configs.values():v.sort(key=lambda x:x[0])
layer=SimpleNamespace(fp32_configs=configs,humming_metas=metas,w13_post_scale=SimpleNamespace(numel=lambda:32),_parameters={},_buffers={});view,manifest=residency_layer_view(layer,4,2,90,'cp_async')
rows=[]
for m in range(1,513):
 default=lambda obj:{s:next(c for lo,hi,c in entries if lo<m*8<=hi) for s,entries in obj.fp32_configs.items()}
 a=ns['choose_partition_configs'](layer,m*8,256);b=ns['choose_partition_configs'](view,m*8,256)
 assert (a is None)==(b is None)
 changed=[]
 if a is None:
  if default(layer)!=default(view):changed.append('default')
 else:
  for arm in ('light','heavy'):
   if a[arm]!=b[arm]:changed.append(arm)
 rows.append({'m':m,'changed_arms':changed})
assert [x['m'] for x in rows if x['changed_arms']==['default']]==list(range(1,15))
assert [x['m'] for x in rows if x['changed_arms']==['light']]==list(range(15,113))
assert not any(x['changed_arms'] for x in rows if x['m']>=113)
r={'status':'actual_helper_CPU_selection_audit_passed','scope_correction':'The blockM8 config view affects default requestsM1–14 AND adaptive light armsM15–112. M15 is not an unchanged fallback boundary. RequestsM113–512 unchanged.','selected_cases':[x for x in rows if x['m'] in (1,8,14,15,32,64,112,113,128,256,512)],'helper_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'limits':'CPU metadata selection only; route-dependent light/heavy populations and native H200 execution remain untested.'}
(P/'RESIDENCY_ADAPTIVE_COVERAGE_CORRECTION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
for f in ('build_residency_hopper.py','prepare_h200_matrix.py'):
 p=P/f;s=p.read_text(encoding='utf-8-sig').replace('[1,8,14,15,128]','[1,8,14,15,32,112,113,128]')
 s=s.replace('M<=14 decode only; retain all other original intervals. TokenM15 checks fallback boundary.','BlockM8 configs affect default requests M1–14 and adaptive light arms M15–112. M113+ is unchanged. Validate both boundaries and inspect selected arms.')
 s=s.replace('M8-only residency candidate B','blockM8 residency candidate B including adaptive light arms M15–112')
 p.write_text(s)
