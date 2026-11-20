from pathlib import Path
import ast,copy,hashlib,json,random
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=P.parents[1]/'hopper_moe'
src=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py';tree=ast.parse(src.read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('select_ep_config','choose_partition_configs')];ns={'deepcopy':copy.deepcopy};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
owned=json.loads((OLD/'grid_body1_retry/owned_configs.json').read_text());layer=SimpleNamespace(fp32_configs=owned['indexed'],w13_post_scale=SimpleNamespace(numel=lambda:32));rows=[]
for m in (8,32,128,256):
 plan=ns['choose_partition_configs'](layer,m*8,256);assert plan
 light=plan['light']['w13'];heavy=plan['heavy']['w13'];cut=plan['cutoff'];entry={'m':m,'cutoff':cut,'light':light,'heavy':heavy,'legal_patterns':[]}
 for pattern in ('uniform','hot_one_local','hot_eight_local'):
  rng=random.Random(318+m)
  if pattern=='uniform':global_ids=[rng.sample(range(256),8) for _ in range(m)]
  elif pattern=='hot_one_local':global_ids=[[0]+rng.sample(range(32,256),7) for _ in range(m)]
  else:global_ids=[rng.sample(range(8),8) for _ in range(m)]
  assert all(len(set(row))==8 for row in global_ids)
  counts=[sum(e in row for row in global_ids) for e in range(32)]
  original_tiles=sum((v+(light if v<=cut else heavy)['block_shape'][0]-1)//(light if v<=cut else heavy)['block_shape'][0] for v in counts)
  original_padded=sum(((v+(light if v<=cut else heavy)['block_shape'][0]-1)//(light if v<=cut else heavy)['block_shape'][0])*(light if v<=cut else heavy)['block_shape'][0] for v in counts)
  candidate_tiles=sum((v+(light['block_shape'][0] if v<=cut else 32)-1)//(light['block_shape'][0] if v<=cut else 32) for v in counts)
  candidate_padded=sum(((v+(light['block_shape'][0] if v<=cut else 32)-1)//(light['block_shape'][0] if v<=cut else 32))*(light['block_shape'][0] if v<=cut else 32) for v in counts)
  compatible=heavy['block_shape'][1:]==[256,128] and heavy['warp_shape'][1:]==[64,64]
  entry['legal_patterns'].append({'pattern':pattern,'global_top8_unique':True,'active_routes':sum(counts),'max_expert_count':max(counts),'original_M_tiles':original_tiles,'M2_M_tiles':candidate_tiles,'requested_weight_ratio':candidate_tiles/original_tiles if original_tiles else 1,'original_padded_rows':original_padded,'M2_padded_rows':candidate_padded,'heavy_K_sequence_compatible_with_current_M2':compatible})
 rows.append(entry)
old=json.loads((P/'sm86_reuse_01/results.json').read_text());old_scope=[]
for row in old['rows']:
 m=row['m'];rng=random.Random(318+m)
 global_rows=[[rng.randrange(256) for _ in range(8)] for _ in range(m)] if row['pattern']=='uniform' else [[0]*8 for _ in range(m)]
 old_scope.append({'m':m,'pattern':row['pattern'],'nonunique_top8_rows':sum(len(set(x))<8 for x in global_rows),'scope':'mechanics stress fixture; fixed BM16 baseline, not actual scored adaptive baseline'})
r={'status':'actual_scored_policy_CPU_audit','helper_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'rows':rows,'prior_receipt_sha256':hashlib.sha256((P/'sm86_reuse_01/results.json').read_bytes()).hexdigest(),'prior_scope_correction':old_scope,'conclusion':'Current M2 has a narrow legal M32 traffic advantage but no requested-weight advantage at M8, M128 or M256 versus scored adaptive policy. M8 actual light uses BK64/WK32, not the fixed BK128/WK64 experiment. Do not promote old stress speedups or run a selective fixed-light comparison as serving evidence.'}
(P/'ACTUAL_SCORED_POLICY_AND_LEGAL_ROUTES.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r,indent=2))
