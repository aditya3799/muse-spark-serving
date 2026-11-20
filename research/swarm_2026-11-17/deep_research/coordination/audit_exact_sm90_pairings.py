"""Exact contiguous pairings; older deduplicated compile list is retained unchanged."""
import copy,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3];OUT=pathlib.Path(__file__).resolve().parent
source=ROOT/'swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json';a=json.loads(source.read_text())
def config(stage,m):return copy.deepcopy(next(c for lo,hi,c in a['stages'][stage]['intervals'] if lo<m<=hi))
segments=[];active={};by_m={}
for m in range(1,16385):
 heavy={s:config(s,8*m) for s in ('w13','w2')};light={s:config(s,m) for s in heavy}
 eligible=any(heavy[s]!=light[s] for s in heavy) and all(light[s]['block_shape'][0]<=heavy[s]['block_shape'][0] for s in heavy)
 for stage in heavy:
  corrected=copy.deepcopy(light[stage]);corrected['block_shape'][2]=heavy[stage]['block_shape'][2];corrected['warp_shape'][2]=heavy[stage]['warp_shape'][2]
  state={'stage':stage,'eligible':eligible,'heavy':heavy[stage],'original_light':light[stage],'K_preserving_light':corrected,'current_large_export_repairs_K':m>512,'changes_K':corrected!=light[stage]}
  key=json.dumps(state,sort_keys=True)
  if stage in active and active[stage][0]==key and active[stage][1]['last_m']==m-1:active[stage][1]['last_m']=m
  else:
   row=dict(state,first_m=m,last_m=m);segments.append(row);active[stage]=(key,row)
 if m in (8,32,128,256,512,513,576,656,657,768,872,873,1016,1017,1024,1360,1361,1536,3072,4096,8192,16384):by_m[m]={'heavy':heavy,'light':light,'eligible':eligible}
report={'status':'EXACT_SOURCE_PAIRING_NOT_RUNTIME_VALIDATION','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'supersedes_for_pairing_lookup':'SM90_K_PRESERVING_POLICIES.json is a deduplicated compile-configuration list; its retained baseline is only the first occurrence and first/last bounds need not define constant heavy policy. Use this exact contiguous report for pairing analysis.','segments':segments,'selected_m':by_m,'source_layout':'token rows M; total top8 route slots8M; expected-local tuningM for32/256experts','new_submissions':0}
(OUT/'SM90_EXACT_POLICY_PAIRINGS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'segments':len(segments),'selected_m_count':len(by_m),'small_K_mismatched_segments':sum(s['eligible'] and s['changes_K'] and s['first_m']<=512 for s in segments)}))
