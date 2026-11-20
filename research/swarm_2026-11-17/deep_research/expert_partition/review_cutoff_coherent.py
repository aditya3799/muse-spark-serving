import collections,hashlib,json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
folder=P/'cutoff_coherent_01';result=folder/'results.json';data=json.loads(result.read_text());groups=collections.defaultdict(list)
for row in data['fullchain']:
 if 'rank' in row:groups[(row['m'],row['pattern'],row['cutoff_factor'])].append(row)
rows=[]
for (m,pattern,factor),group in sorted(groups.items()):
 if len(group)!=8:continue
 assert {v['rank'] for v in group}==set(range(8))
 assert len({v['global_routes_sha256'] for v in group})==1
 assert sum(v['active_slots'] for v in group)==m*8
 for v in group:
  v['pooled_A']=statistics.median(v['samples_us']['A1']+v['samples_us']['A2']);v['pooled_B']=statistics.median(v['samples_us']['B1']+v['samples_us']['B2'])
 baseline=max(group,key=lambda x:x['pooled_A']);candidate=max(group,key=lambda x:x['pooled_B'])
 rows.append({'tokens':m,'pattern':pattern,'cutoff_factor':factor,'global_route_hash':group[0]['global_routes_sha256'],'sum_local_slots':sum(v['active_slots'] for v in group),'all_rank_controls_pass':all(v['noise_pass'] for v in group),'failed_control_ranks':[v['rank'] for v in group if not v['noise_pass']],'baseline_max_rank':baseline['rank'],'candidate_max_rank':candidate['rank'],'baseline_max_us':baseline['pooled_A'],'candidate_max_us':candidate['pooled_B'],'max_rank_speedup_proxy':baseline['pooled_A']/candidate['pooled_B'],'rank_rows':[{'rank':v['rank'],'active_slots':v['active_slots'],'A_us':v['pooled_A'],'B_us':v['pooled_B'],'speedup':v['speedup'],'controls':v['controls'],'noise_pass':v['noise_pass']} for v in group]})
report={'scope':'Sequential SM86 rank-cost surrogate with same canonical weights. Includes no real rank concurrent execution, collectives, overlap, distinctrankweights, H200 or serving claim. Only all-control-passing groups qualify their local max-rank proxy.','run_status':data['status'],'report_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'rows':rows}
(P/'CUTOFF_COHERENT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:v for k,v in row.items() if k!='rank_rows'} for row in rows],indent=2))
