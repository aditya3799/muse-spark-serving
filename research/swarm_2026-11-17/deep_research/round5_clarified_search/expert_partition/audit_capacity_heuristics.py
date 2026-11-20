import pathlib,json,hashlib,dataclasses,ast
P=pathlib.Path(__file__).resolve().parent;R=next(p for p in P.parents if (p/'RULES.md').exists());audit=R/'swarm_2026-11-16/moe/audit_sm90_native.py'
prefix=audit.read_text(encoding="utf-8-sig").split("report={'scope':",1)[0];ns={'__file__':str(audit)};exec(compile(prefix,str(audit),'exec'),ns)
config,d,enums,sm90=(ns[n]for n in ['config','d','enums','sm90']);vendor=ns['vendor'];rows={}
for capacity in [32,33,40,64]:
 rows[capacity]={}
 for stage,n in [('w13',8192),('w2',4096)]:
  meta=config.LayerConfig(shape_n=n,shape_k=4096,num_experts=capacity,a_dtype=d.bfloat16,b_dtype=d.float4e2m1,c_dtype=d.bfloat16,bs_dtype=d.float8e4m3,weight_scale_group_size=16,weight_scale_type=enums.WeightScaleType.GROUP,weight_scale_2_type=enums.WeightScale2Type.NONE)
  intervals=sm90.Sm90Heuristics.get_configs(meta,use_f16_accum=False,gemm_type=enums.GemmType.INDEXED)
  for lo,hi,c in intervals:c['use_stream_k']=False
  rows[capacity][stage]={'meta':json.loads(meta.to_str()),'intervals':intervals}
comparisons=[]
for cap in [33,40,64]:
 for stage in ['w13','w2']:
  b=rows[32][stage]['intervals'];c=rows[cap][stage]['intervals'];changed=[]
  for m in [8,32,64,128,256,768,1024,3012,4096,8192,16384]:
   bc=next(v for lo,hi,v in b if lo<m<=hi);cc=next(v for lo,hi,v in c if lo<m<=hi)
   if bc!=cc:changed.append({'routes':m,'original':bc,'capacity_derived':cc})
  comparisons.append({'capacity':cap,'stage':stage,'all_intervals_identical':b==c,'sample_changes':changed})
report={'status':'SOURCE_ONLY_REAL_SM90_HEURISTICS_CAPACITY_SWEEP','capacities':rows,'comparisons':comparisons,'source_hashes':ns['hashes'],'scope':'Actual pinned heuristic/LayerConfig with mocked architecture; no compile/GPU. Changing E affects heuristic routing-distribution estimator; preserve/pin original32 configs for causal slot-copy probe.'}
(P/'CAPACITY_HEURISTICS_SOURCE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'comparisons':[{k:v for k,v in x.items()if k!='sample_changes'}|{'sample_changes':len(x['sample_changes'])}for x in comparisons]}))
