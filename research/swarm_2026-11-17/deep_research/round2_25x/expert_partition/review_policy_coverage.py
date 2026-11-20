import ast,copy,hashlib,json,pathlib,types
p=pathlib.Path(__file__).resolve().parent;root=p.parents[3];old=root/'swarm_2026-11-17/deep_research/expert_partition';scored=root/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py';fixed=old/'scalable_partition.py'
ns={'deepcopy':copy.deepcopy};tree=ast.parse(scored.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('select_ep_config','choose_partition_configs')];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(scored),'exec'),ns);original=ns['choose_partition_configs'];ns2={'deepcopy':copy.deepcopy,'original_choose_partition_configs':original};tree=ast.parse(fixed.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='choose_partition_configs'];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(fixed),'exec'),ns2)
sm86=old/'k_preserve_m1024_01/owned_configs.json';sm90=root/'swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json';tables={'SM86':json.loads(sm86.read_text())['indexed'],'SM90':{s:v['intervals'] for s,v in json.loads(sm90.read_text())['stages'].items()}};rows=[];noop_intervals=[]
def assess(arch,table,m):
 layer=types.SimpleNamespace(fp32_configs=table,w13_post_scale=types.SimpleNamespace(numel=lambda:32));r=m*8;before=original(layer,r,256);after=ns2['choose_partition_configs'](layer,r,256)
 row={'arch':arch,'m':m,'routes':r,'scored_helper_path':'adaptive' if r<=4096 else 'default_due_cap','repaired_plan_available':after is not None,'original_selection_available':before is not None}
 if after is None:
  default={s:next(c for lo,hi,c in intervals if lo<r<=hi) for s,intervals in table.items()}
  selected={s:ns['select_ep_config'](intervals,r,32,256) for s,intervals in table.items()}
  light={s:value['config'] for s,value in selected.items() if value is not None}
  row['rejection_analysis']={'selection_missing':len(light)!=2,'all_original_configs_equal':light==default,'light_row_tile_no_larger':all(light[s]['block_shape'][0]<=default[s]['block_shape'][0] for s in light),'light_configs':light,'default_configs':default}
 if after:
  row.update(cutoff=after['cutoff'],all_stages_light_equal_heavy=all(after['light'][s]==after['heavy'][s] for s in ('w13','w2')),stages={})
  for s in ('w13','w2'):
   l,h=after['light'][s],after['heavy'][s];row['stages'][s]={'light_block':l['block_shape'],'heavy_block':h['block_shape'],'light_warp':l['warp_shape'],'heavy_warp':h['warp_shape'],'equal_after_repair':l==h,'K_repaired':before['light'][s]['block_shape'][2]!=l['block_shape'][2] or before['light'][s]['warp_shape'][2]!=l['warp_shape'][2],'remaining_different_fields':[k for k in sorted(set(l)|set(h)) if l.get(k)!=h.get(k)],'whole_light':l,'whole_heavy':h}
 return row
for arch,table in tables.items():
 for m in (513,753,768,1024,2048,3012,4096,8192,16384):rows.append(assess(arch,table,m))
 start=None
 for m in range(513,16385):
  row=assess(arch,table,m);noop=row.get('all_stages_light_equal_heavy',False)
  if noop and start is None:start=m
  if not noop and start is not None:noop_intervals.append({'arch':arch,'first_m':start,'last_m':m-1});start=None
 if start is not None:noop_intervals.append({'arch':arch,'first_m':start,'last_m':16384})
report={'status':'CPU_NATIVE_POLICY_COVERAGE_SWEEP','source_sha256':{str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (scored,fixed,sm86,sm90)},'geometry':{'H':4096,'I':4096,'local_E':32,'global_E':256,'topk':8},'rows':rows,'no_op_plan_contiguous_intervals':noop_intervals,'limits':['Scored chooser alone can return plans above4096 routes, but scored helper guards before calling it; scored runtime remains default there','Scalable chooser restores heavy K topology after original distinct test and does not recheck full equality','SM90 table architecture-mocked source only, not runtime proof','All M above4096 tokens are untested by current composed numerical gates; large allocation not performed','Equal row tiles do not imply equal arithmetic/configuration; full dict equality is used for no-op detection','SM86 num_sms20 retained consistently; SM90 minimal configs use pinned defaults']}
(p/'NATIVE_POLICY_COVERAGE_SWEEP.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'noops':noop_intervals,'rows':[{k:row.get(k) for k in ('arch','m','cutoff','repaired_plan_available','all_stages_light_equal_heavy')} for row in rows]}))

