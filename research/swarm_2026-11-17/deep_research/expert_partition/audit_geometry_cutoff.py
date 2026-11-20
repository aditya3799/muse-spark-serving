import ast,copy,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
source=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
tree=ast.parse(source.read_text());ns={'deepcopy':copy.deepcopy};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('select_ep_config','choose_partition_configs')],type_ignores=[]),'<scored>','exec'),ns)
ns['original_choose_partition_configs']=ns['choose_partition_configs']
tree=ast.parse((P/'geometry_cutoff_partition.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='choose_partition_configs' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_base_k_preserving_chooser' for t in n.targets)]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<geometry>','exec'),ns)
paths={'SM86':P/'k_preserve_m1024_01/owned_configs.json','SM90':ROOT/'swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json'};out={}
for arch,path in paths.items():
 data=json.loads(path.read_text());cfg=data['indexed'] if arch=='SM86' else {s:v['intervals'] for s,v in data['stages'].items()};snapshot=copy.deepcopy(cfg);layer=SimpleNamespace(fp32_configs=cfg,w13_post_scale=SimpleNamespace(numel=lambda:32));changed=[];intervals=[];oldkey=None
 for m in range(1,16385):
  a=ns['_base_k_preserving_chooser'](layer,m*8,256);b=ns['choose_partition_configs'](layer,m*8,256)
  if m<=512:assert a==b
  if a is None:assert b is None;key=None
  else:
   assert a['light']==b['light'] and a['heavy']==b['heavy']
   doubled=b['cutoff']==a['cutoff']*2
   assert doubled or b['cutoff']==a['cutoff']
   if doubled:
    changed.append(m)
    for stage in ('w13','w2'):
     l=a['light'][stage]['block_shape'][0];h=a['heavy'][stage]['block_shape'][0]
     for c in range(a['cutoff']+1,b['cutoff']+1):assert math.ceil(c/l)*l<math.ceil(c/h)*h
   key=(doubled,a['cutoff'],tuple((s,a['light'][s]['block_shape'][0],a['heavy'][s]['block_shape'][0]) for s in ('w13','w2')))
  if key==oldkey and intervals:intervals[-1]['last_M']=m
  else:intervals.append({'first_M':m,'last_M':m,'description':key});oldkey=key
 assert cfg==snapshot
 out[arch]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'token_rows_checked':16384,'changed_token_rows':len(changed),'contiguous_intervals':intervals,'small_path_unchanged':True,'all_new_light_assignments_have_strictly_less_padded_rows_in_both_stages':True,'all_gemm_configs_unchanged':True}
report={'scope':'CPU metadata/selection proof only. Strictly less padded rows does not prove faster latency or H200 qualification; N-shape/weight reuse/occupancy absent. Export unchanged.','candidate_sha256':hashlib.sha256((P/'geometry_cutoff_partition.py').read_bytes()).hexdigest(),'architectures':out}
(P/'GEOMETRY_CUTOFF_CPU_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{n:v for n,v in a.items() if n!='contiguous_intervals'} for k,a in out.items()},indent=2))
