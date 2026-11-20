import hashlib,json,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[3]
audit=json.loads((p/'SM90_OCCUPANCY_REVIEW.json').read_text());meta_source=root/'swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json';meta=json.loads(meta_source.read_text())['stages']['w13']['meta'];jobs=[];checks=[]
for row in audit['rows']:
 if row['pattern']!='uniform':continue
 for kind,cfg in row['policy']['w13'].items():
  bm,bn,bk=cfg['block_shape'];wm=cfg['warp_shape'][0];actual_splits=cfg.get('num_write_splits',1);h=4096
  job={'name':'w13_M%d_%s'%(row['m'],kind),'tokens':row['m'],'policy':kind,'meta':meta,'tuning_config':cfg,'compute_config':{'gemm_type':'indexed','use_f16_accum':False},'sm_version':90,'expected_num_write_splits':actual_splits,'eligibility':'Root runtime launcher refuses SM90; compile-only use rendered epilogue with explicit SM90 adapter and full original config'}
  jobs.append(job)
  # Validate both address write splits when algorithmically defined. Actual policy uses1.
  for splits in (1,2):
   if splits==2 and (bm%32 or bm!=wm):continue
   seen=set()
   for split in range(splits):
    for offset in range(bm*bn//8//splits):
     sr,sc=divmod(offset,8);gr=sr%(bm//splits)+(bm//2*split if splits==2 else 0);gc=sr//(bm//splits)*8+sc
     for nb in range(2*h//bn):
      index=nb*(bn//8)+gr*(2*h//8)+gc
      for pair in range(4):
       output=index*4+pair
       assert output==gr*h+nb*(bn//2)+gc*4+pair
       assert output not in seen;seen.add(output)
   assert seen==set(range(bm*h))
   checks.append({'job':job['name'],'BM':bm,'BN':bn,'BK':bk,'warp_M':wm,'write_splits':splits,'actual_policy':splits==actual_splits,'unique_output_addresses':len(seen),'gate_up_permutation':'pair2j maps original gate j and up j+I','passed':True})
report={'status':'FOUR_NATIVE_W13_CPU_LAYOUT_PASS_COMPILE_PENDING','source_sha256':hashlib.sha256(meta_source.read_bytes()).hexdigest(),'jobs':jobs,'layout_checks':checks,'limits':['CPU address bijection only; no GPU shared-memory arithmetic validation','No compiler executed','num_write_splits2 only tested where pinned tune/base.py legality BM==warpM and BM divisible32 permits it','All actual four policies use num_write_splits1; extra split2 cases are structural supplements','SM90 runtime remains refused by original fused launcher; no eligibility assertion bypass in serving code']}
(p/'SM90_COMPOSED_W13_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'jobs':[j['name'] for j in jobs],'layout_checks':checks}))
