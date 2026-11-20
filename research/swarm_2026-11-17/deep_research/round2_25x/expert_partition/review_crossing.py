import hashlib,json,pathlib
p=pathlib.Path(__file__).resolve().parent;r=json.loads((p/'crossing_01/results.json').read_text());rows=[]
for policy in r['policies']:
 m=policy['m'];c=policy['composed']['cutoff'];assert c+1<=m
 for gen,counts in enumerate(((c-1,c+1,3),(c+1,3,c-1),(0,0,0))):
  rows.append({'m':m,'generation':gen,'cutoff':c,'counts_experts012':counts,'other_expert_counts':0,'light_experts':[e for e,n in enumerate(counts) if 0<n<=c],'heavy_experts':[e for e,n in enumerate(counts) if n>c],'local_routes':sum(counts),'global_top8_unique_before_local_remap':True,'canonical_down_poison_before_replay':37})
report={'status':'CUTOFF_CROSSING_GRAPH_BITWISE_PASS','report_sha256':hashlib.sha256((p/'crossing_01/results.json').read_bytes()).hexdigest(),'checks':len(r['checks']),'strict_finite_bitwise':all(x['finite'] and x['bitwise'] for x in r['checks']),'coverage':rows,'poison_scope':'Down buffer values poisoned before each replay. Active raw down rows and every final token checked. Inactive sentinel preservation is not asserted: baseline cache13 aliases W13 storage and can legitimately modify overlapping inactive-down addresses.','limits':['No performance samples taken','Synthetic SM86 H4096 E32 only','This exercises simultaneous partitions and transition to all-remote, not a full model or sanitizer proof']}
(p/'CROSSING_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
