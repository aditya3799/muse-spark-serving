from pathlib import Path
import json,math,hashlib
P=Path(__file__).resolve().parent;OLD=P.parents[2]/'swarm_2026-11-16/moe/phase4_hour'
source=json.loads((OLD/'producer_compile_jobs.json').read_text())['jobs'];resources=[]
for f in ('producer_all_00','producer_all_22','producer_all_44','producer_all_66'):
 resources.extend(json.loads((OLD/f/'results.json').read_text())['cases'])
rows=[]
for m in (1,8,14,15,32,128,256,512,1024):
 for stage in ('w13','w2'):
  job=next(j for j in source if j['stage']==stage and j['policy']=='tma_b' and any(lo<m*8<=hi for lo,hi in j['intervals']))
  conf=job['baseline'];bm,bn,bk=conf['block_shape'];wm,wn,wk=conf['warp_shape']
  nwarps=bn//64;kparts=bk//wk;threads=(bm//wm)*nwarps*kparts*32
  resource=next(r for r in resources if r['stage']==stage and r['policy']=='tma_b' and r['configuration']['block_shape']==conf['block_shape'])
  shared=resource['shared_bytes_estimated'];accum=wm*64//32
  row={'tokens':m,'stage':stage,'original_block':conf['block_shape'],'original_warp':conf['warp_shape'],'proposed_warp':[wm,64,wk],'K_partition_count_unchanged':kparts,'N_warps':nwarps,'warpgroup_constraint_pass':nwarps%4==0,'proposed_math_threads':threads,'original_shared_bytes_stage4':shared,'two_CTA_shared_capacity_pass':2*shared<=228*1024,'minimum_FP32_output_accumulator_registers_per_thread':accum,'CTA2_register_capacity_model_per_thread':min(255,65536//(2*threads)),'mandatory_spill_from_accumulators_alone':accum>255,'native_shape_preserved':[wm,64,16],'numerics':'Requires native Hopper mapping and bitwise gates; same primitive is necessary, not sufficient'}
  rows.append(row)
r={'status':'CPU_static_configuration_eligibility_only','source_sha256':hashlib.sha256((OLD/'producer_compile_jobs.json').read_bytes()).hexdigest(),'scope':'Pinned full-route default configurations selected by tokenM*8; adaptive policies must be enumerated separately. No scheduling policy enabled.','assumptions':['Changing warpN only leaves block, warpK, K-partition count, staged A/B/scale footprint and per-output K16 sequence unchanged','Minimum output registers assumes normal FP32 accumulator representation; excludes current/next B operands, quant/scale registers, descriptors and compiler temporaries','CTA register capacity and shared capacity are necessary constraints, not runtime occupancy','Mma native orientation is64weightNxlogicaltokenM; no assumed64token padding'],'rows':rows}
(P/'WARP_N64_BROAD_ELIGIBILITY.json').write_text(json.dumps(r,indent=2)+'\n')
for row in rows:
 if row['tokens'] in (32,128,256):print(row['tokens'],row['stage'],'block',row['original_block'],'threads',row['proposed_math_threads'],'warpgroups',row['warpgroup_constraint_pass'],'shared',row['original_shared_bytes_stage4'],'Cregs',row['minimum_FP32_output_accumulator_registers_per_thread'],'CTA2limit',row['CTA2_register_capacity_model_per_thread'])
