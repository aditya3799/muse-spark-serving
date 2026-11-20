"""Native-only policy/layout guard audit; imports neither Torch nor Humming."""
import ast,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
OWN=Path(__file__).resolve().parent;MOE=OWN.parent/'moe';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def extract(path,name,klass=None):
 t=ast.parse(path.read_text(encoding='utf-8-sig'))
 if klass:t=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==klass)
 return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
ns={};exec(compile((OWN/'ep_tuning_selector.py').read_text(),'<selector>','exec'),ns)
exec(compile(ast.Module(body=[extract(OWN/'adaptive_expert_split.py','choose_partition_configs')],type_ignores=[]),'<planner>','exec'),ns)
kns={'MmaType':SimpleNamespace(WGMMA='wgmma'),'jit_utils':SimpleNamespace(is_power_of_two=lambda x:x>0 and x&(x-1)==0)}
kernel=MOE/'humming_vendor/humming/kernel/humming.py';exec(compile(ast.Module(body=[extract(kernel,'check_shape','HummingKernel')],type_ignores=[]),str(kernel),'exec'),kns)
audit=json.loads((MOE/'sm90_wgmma_metadata_audit.json').read_text());base=SimpleNamespace(w13_post_scale=SimpleNamespace(numel=lambda:32),fp32_configs={s:v['intervals'] for s,v in audit['stages'].items()});snapshot=json.dumps(base.fp32_configs)
ms=set(range(1,513));max_m=(1<<30)//8
for entry in audit['stages'].values():
 for lo,hi,c in entry['intervals']:
  for boundary in (lo,hi):
   for factor in (1,8):
    pivot=int(boundary)//factor
    ms.update(m for m in (pivot-1,pivot,pivot+1) if 1<=m<=max_m)
rows=[];shape_checks=0
for m in sorted(ms):
 plan=ns['choose_partition_configs'](base,m*8,256)
 eligible=m*8<=4096 and plan is not None
 if plan:
  for stage in ('w13','w2'):
   meta=audit['stages'][stage]['meta']
   for arm in ('light','heavy'):
    c=plan[arm][stage]
    obj=SimpleNamespace(problem_shape=(0,meta['shape_n'],meta['shape_k']),pad_shape=(0,0,0),block_shape=c['block_shape'],warp_shape=c['warp_shape'],a_dtype=SimpleNamespace(num_bits=16),mma_type='wgmma')
    kns['check_shape'](obj);assert c['use_stream_k'] is False and c['use_f16_accum'] is False
    assert math.prod(c['block_shape'])//math.prod(c['warp_shape'])*32<=1024 and c['num_stages']>=3
    shape_checks+=1
  light=plan['light'];heavy=plan['heavy'];alignments=len({(arm,tuple(plan[arm][stage]['block_shape'])[0]) for arm in ('light','heavy') for stage in ('w13','w2')})
  entry={'m':m,'actual_routes':m*8,'eligible':eligible,'fallback_reason':None if eligible else 'route count exceeds bounded4096 path','cutoff':plan['cutoff'],'light_blocks':{s:light[s]['block_shape'] for s in light},'heavy_blocks':{s:heavy[s]['block_shape'] for s in heavy},'alignment_calls':alignments}
 else:entry={'m':m,'actual_routes':m*8,'eligible':False,'fallback_reason':'identical policy or larger expected-local row tile'}
 rows.append(entry)
assert json.dumps(base.fp32_configs)==snapshot
out={'status':'PASS_CPU_SM90_POLICY_LAYOUT_GUARDS','scope':'All512 eligible positive M values (topk8) plus every metadata interval boundary under default and expected-local selection, and neighbors; no CUDA compilation or hardware execution','metadata_intervals_per_stage':{s:len(v['intervals']) for s,v in audit['stages'].items()},'tested_m_count':len(rows),'shape_guard_checks':shape_checks,'eligible_split_m_count':sum(r['eligible'] for r in rows),'max_alignment_calls_eligible':max(r.get('alignment_calls',0) for r in rows if r['eligible']),'representative':[r for r in rows if r['m'] in (1,8,33,128,512,513,16384)],'rows':rows,'packing_proof':'Weight and scale transformation uses layer dtype/quantization/MMA/layout metadata, not tuning block/warp shape. Both arms use identical layer metadata/canonical pointers; existing indexed output layout maps original slots. Different blockK still spans full shapeK with Stream-K off.','limits':['SM90 M33/M128 pairs differ from SM86; local speed results do not transfer.','At most4096 routes and256experts enter split path; larger serving batches use exact default helper.','Stage blockM differences are covered by alignment cache keyed(partition,blockM), yielding up to4alignments.','Kernel launch/resource/precision behavior still needs actual SM90 compile/runtime checks.','Added launches may dominate Hopper latency.','CPU grid/partition guards are not whole-model memory evidence.'],'hashes':{'prototype':sha(OWN/'adaptive_expert_split.py'),'selector':sha(OWN/'ep_tuning_selector.py'),'audit':sha(MOE/'sm90_wgmma_metadata_audit.json'),'humming_kernel':sha(kernel)}}
(OWN/'ADAPTIVE_SPLIT_SM90_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('rows','limits','hashes')}))
