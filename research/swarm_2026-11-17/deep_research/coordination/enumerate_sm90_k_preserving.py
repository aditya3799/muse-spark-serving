import copy,hashlib,json,pathlib
P=pathlib.Path(r'D:\OpenKernelChallenge\serving')
O=P/'swarm_2026-11-17/deep_research/coordination'
S=P/'swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json'
a=json.loads(S.read_text());seen={};rows=[]
def cfg(stage,m):
 return copy.deepcopy(next(c for lo,hi,c in a['stages'][stage]['intervals'] if lo<m<=hi))
for m in range(513,16385):
 configs={s:(cfg(s,8*m),cfg(s,m)) for s in ('w13','w2')}
 if not any(h!=l for h,l in configs.values()) or not all(l['block_shape'][0]<=h['block_shape'][0] for h,l in configs.values()): continue
 for stage,(heavy,light) in configs.items():
  original=copy.deepcopy(light)
  light['block_shape'][2]=heavy['block_shape'][2]
  light['warp_shape'][2]=heavy['warp_shape'][2]
  key=stage+json.dumps(light,sort_keys=True)
  if key in seen:seen[key]['last_token_rows']=m;continue
  threads=32
  divisible=True
  for b,w in zip(light['block_shape'],light['warp_shape']):
   divisible &= b%w==0
   threads*=b//w
  r={'stage':stage,'first_token_rows':m,'last_token_rows':m,'original_light':original,'baseline':heavy,'corrected_light':light,'k_changed':original['block_shape'][2]!=light['block_shape'][2] or original['warp_shape'][2]!=light['warp_shape'][2],'static_thread_count':threads,'dimensions_divisible':divisible,'thread_limit_pass':threads<=1024,'configuration':dict(a['stages'][stage]['meta'],**light,gemm_type='indexed',use_cp_async=True)}
  seen[key]=r;rows.append(r)
report={'status':'SOURCE_POLICY_ENUMERATION_NOT_COMPILER_OR_GPU_VALIDATION','source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'rows':rows,'all_divisible':all(r['dimensions_divisible'] for r in rows),'all_thread_limit_pass':all(r['thread_limit_pass'] for r in rows),'remaining':['Native policy instantiation and generated-kernel shape assertions','OfflineSM90 compile and sharedmemory/register resources','H200 raw/final numerical equality and changedinput graph replay','Fullmodel TTFT/memory/serving']}
(O/'SM90_K_PRESERVING_POLICIES.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'unique_stage_configs':len(rows),'changed_K':sum(r['k_changed'] for r in rows),'all_divisible':report['all_divisible'],'all_thread_limit_pass':report['all_thread_limit_pass'],'rows':[{k:r[k] for k in ('stage','first_token_rows','last_token_rows','k_changed','static_thread_count')} for r in rows]},indent=2))
