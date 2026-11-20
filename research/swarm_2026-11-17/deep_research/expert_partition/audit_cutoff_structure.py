import ast,collections,importlib.util,json,math,random
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
spec=importlib.util.spec_from_file_location('tile_study',ROOT/'optimization_marlin_tiles_2026-11-15/tile_study.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tree=ast.parse((P/'probe_cutoff_coherent.py').read_text(encoding='utf-8-sig'));main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');fn=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='make_routes');ns={'random':random,'stock_make_routes':module.make_routes};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<route-fixture>','exec'),ns)
configs=json.loads((P/'k_preserve_m1024_01/owned_configs.json').read_text())['indexed']
select=lambda stage,n:next(c for lo,hi,c in configs[stage] if lo<n<=hi)
rows=[]
for m in (768,1024):
 light={s:select(s,m)['block_shape'][0] for s in configs};heavy={s:select(s,m*8)['block_shape'][0] for s in configs}
 for pattern in ('uniform','hot_rank','mixed'):
  global_ids=ns['make_routes'](m,8,256,8,pattern,216+m)
  for rank in range(8):
   counts=collections.Counter(v-rank*32 for ids in global_ids for v in ids if rank*32<=v<(rank+1)*32)
   for factor in (1,2,3):
    cutoff=min(light.values())*factor
    stages={}
    for stage in configs:
     stages[stage]={'baseline_row_tiles':sum(math.ceil(c/heavy[stage]) for c in counts.values()),'candidate_row_tiles':sum(math.ceil(c/(light[stage] if c<=cutoff else heavy[stage])) for c in counts.values()),'baseline_padded_rows':sum(math.ceil(c/heavy[stage])*heavy[stage] for c in counts.values()),'candidate_padded_rows':sum(math.ceil(c/(light[stage] if c<=cutoff else heavy[stage]))*(light[stage] if c<=cutoff else heavy[stage]) for c in counts.values())}
    rows.append({'m':m,'pattern':pattern,'rank':rank,'factor':factor,'cutoff':cutoff,'local_slots':sum(counts.values()),'local_experts_active':len(counts),'local_max_count':max(counts.values(),default=0),'heavy_experts':sum(c>cutoff for c in counts.values()),'stages':stages})
report={'scope':'Structural padding/rowtile counts only. Rowtile count is a weight-load opportunity proxy, not measured bytes or performance. No H200 forecast.','rows':rows};(P/'CUTOFF_STRUCTURAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([v for v in rows if v['pattern']=='uniform' and v['rank']==0],indent=2))
