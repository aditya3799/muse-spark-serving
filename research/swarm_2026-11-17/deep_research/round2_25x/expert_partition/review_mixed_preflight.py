import collections,hashlib,json,pathlib,random
p=pathlib.Path(__file__).resolve().parent;r=json.loads((p/'composition_03/results.json').read_text());sm90=json.loads((p/'SM90_OCCUPANCY_REVIEW.json').read_text());rows=[]
for m in (768,1024):
 local=next(v['composed'] for v in r['policies'] if v['m']==m);native=next(v for v in sm90['rows'] if v['m']==m and v['pattern']=='uniform');policies={'SM86':local,'SM90':{'cutoff':native['cutoff'],'light':{s:c['light'] for s,c in native['policy'].items()},'heavy':{s:c['heavy'] for s,c in native['policy'].items()}}}
 for generation in range(3):
  rng=random.Random(1641+m+generation);global_ids=[rng.sample([(generation*8+j)%32 for j in range(8)] if token<m//2 else range(256),8) for token in range(m)]
  assert all(len(set(row))==8 for row in global_ids);counts=[sum(e in row for row in global_ids) for e in range(32)]
  for arch,plan in policies.items():
   c=plan['cutoff'];nl=sum(0<v<=c for v in counts);nh=sum(v>c for v in counts);assert nl>0 and nh>0
   stage_rows={}
   for stage in ('w13','w2'):
    lm=plan['light'][stage]['block_shape'][0];hm=plan['heavy'][stage]['block_shape'][0];base=sum(((v+hm-1)//hm)*hm for v in counts);candidate=sum(((v+(lm if v<=c else hm)-1)//(lm if v<=c else hm))*(lm if v<=c else hm) for v in counts)
    stage_rows[stage]={'light_m':lm,'heavy_m':hm,'baseline_padded_rows':base,'partition_padded_rows':candidate,'padding_ratio':base/candidate}
   rows.append({'m':m,'generation':generation,'arch_policy':arch,'cutoff':c,'global_top8_unique':True,'light_experts':nl,'heavy_experts':nh,'local_routes':sum(counts),'counts':counts,'global_route_sha256':hashlib.sha256(json.dumps(global_ids,separators=(',',':')).encode()).hexdigest(),'stages':stage_rows})
report={'status':'MIXED_SKEW_CPU_PREFLIGHT_NOT_EXECUTED','rows':rows,'scope':'Half tokens select all eight hot local experts; other half select unique global top8. Hot set rotates per graph generation. Synthetic skew fixture, not measured router population; H200 values are policy arithmetic only.'}
(p/'MIXED_SKEW_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:v[k] for k in ('m','arch_policy','local_routes','light_experts','heavy_experts','stages')} for v in rows if v['generation']==0]))
