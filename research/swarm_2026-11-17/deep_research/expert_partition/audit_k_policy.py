import ast,copy,json,hashlib,math
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
base=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
ns={'deepcopy':copy.deepcopy}
t=ast.parse(base.read_text());nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('select_ep_config','choose_partition_configs')];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(base),'exec'),ns)
ns['original_choose_partition_configs']=ns['choose_partition_configs']
p=P/'scalable_partition.py';t=ast.parse(p.read_text(encoding='utf-8-sig'));nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='choose_partition_configs'];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
configs=json.loads((P/'k_preserve_m1024_01/owned_configs.json').read_text())['indexed']
layer=SimpleNamespace(w13_post_scale=SimpleNamespace(numel=lambda:32),fp32_configs=configs)
routes={1,4095,4096,4097,131072}
for intervals in configs.values():
 for lo,hi,c in intervals:
  for x in (lo,hi):
   if math.isfinite(x):
    for factor in (1,2,4,8):
     for offset in (-1,0,1):
      value=int(x)*factor+offset
      if 1<=value<=131072:routes.add(value)
rows=[]
for global_e in (32,64,128,256):
 for actual in sorted(routes):
  before=copy.deepcopy(layer.fp32_configs)
  original=ns['original_choose_partition_configs'](layer,actual,global_e)
  revised=ns['choose_partition_configs'](layer,actual,global_e)
  assert layer.fp32_configs==before
  if actual<=4096:assert original==revised
  if revised is not None and actual>4096:
   for stage in ('w13','w2'):
    l,h=revised['light'][stage],revised['heavy'][stage]
    assert l['block_shape'][2]==h['block_shape'][2] and l['warp_shape'][2]==h['warp_shape'][2]
    assert l['use_f16_accum'] is False and l['use_stream_k'] is False
    assert all(b%w==0 for b,w in zip(l['block_shape'],l['warp_shape']))
    threads=math.prod(l['block_shape'])//math.prod(l['warp_shape'])*32
    assert 32<=threads<=1024 and threads%32==0
  rows.append({'global_experts':global_e,'route_slots':actual,'plan':revised is not None,'input_configs_unchanged':True,'small_path_unchanged':actual<=4096})
report={'scope':'CPU selector and basic thread-shape audit using observed SM86 fixture policies; not SM90 layout/resource qualification','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checks':len(rows),'passed':True,'rows':rows}
(P/'K_POLICY_CPU_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(rows),'passed':True}))
