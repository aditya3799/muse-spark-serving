import copy,hashlib,json,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[3]
source=root/'swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json';table=json.loads(source.read_text());occ=json.loads((p/'TIMING_OCCUPANCY_REVIEW.json').read_text())['occupancy'];rows=[]
for route in occ:
 m=route['m'];r=m*8;policies={}
 for stage,data in table['stages'].items():
  light=copy.deepcopy(next(c for lo,hi,c in data['intervals'] if lo<m<=hi));heavy=copy.deepcopy(next(c for lo,hi,c in data['intervals'] if lo<r<=hi))
  light['block_shape'][2]=heavy['block_shape'][2];light['warp_shape'][2]=heavy['warp_shape'][2]
  policies[stage]={'light':light,'heavy':heavy}
 cutoff=min(v['light']['block_shape'][0] for v in policies.values());counts=route['counts'];nl=sum(0<n<=cutoff for n in counts);nh=sum(n>cutoff for n in counts)
 stage_rows={}
 for stage,policy in policies.items():
  lm=policy['light']['block_shape'][0];hm=policy['heavy']['block_shape'][0]
  baseline=sum(((n+hm-1)//hm)*hm for n in counts)
  candidate=sum(((n+(lm if n<=cutoff else hm)-1)//(lm if n<=cutoff else hm))*(lm if n<=cutoff else hm) for n in counts)
  stage_rows[stage]={'light_m':lm,'heavy_m':hm,'baseline_padded_rows':baseline,'candidate_padded_rows':candidate,'padded_row_ratio':baseline/candidate if candidate else None}
 rows.append({'m':m,'pattern':route['pattern'],'same_route_sha256':route['route_sha256'],'cutoff':cutoff,'light_experts':nl,'heavy_experts':nh,'light_routes':sum(n for n in counts if n<=cutoff),'heavy_routes':sum(n for n in counts if n>cutoff),'policy':policies,'stages':stage_rows})
report={'status':'PINNED_SM90_SOURCE_POLICY_OCCUPANCY_MODEL','source':str(source.relative_to(root)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'geometry':{'H':4096,'I':4096,'local_experts':32,'global_experts':256,'topk':8},'rows':rows,'limits':['Architecture-mocked pinned Humming WGMMA heuristic table; not H200 runtime timing','Exactly same generation0 routes as local composition_03; no fabricated counts','Expected-local selector tune_m=ceil(R*32/256)=M; heavy selector uses R=8M','Large-M repair copies heavy blockK/warpK into light; cutoff remains minimum light BM','Padded row ratios omit N tiles, K stages, warp topology, memory traffic and overhead; cannot be converted to speedup','H200 numerical and end-to-end serving gates remain required']}
(p/'SM90_OCCUPANCY_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:row[k] for k in ('m','pattern','cutoff','light_experts','heavy_experts','stages')} for row in rows]))
