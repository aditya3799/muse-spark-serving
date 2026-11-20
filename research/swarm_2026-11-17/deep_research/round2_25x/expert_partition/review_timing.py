import collections,hashlib,json,pathlib,random,statistics
p=pathlib.Path(__file__).resolve().parent;r=json.loads((p/'composition_03/results.json').read_text());occupancy=[]
for policy in r['policies']:
 m=policy['m'];cutoff=policy['composed']['cutoff']
 for pattern in ('uniform','hot_rank','tail_invalid','all_remote'):
  rng=random.Random(1641+m);ids=[]
  for token in range(m):
   pool=range(32) if pattern=='hot_rank' else range(256)
   if pattern=='all_remote':pool=range(32,256)
   row=rng.sample(pool,8)
   if pattern=='tail_invalid' and token>=m//2:row=[-1]*8
   ids.append([v if 0<=v<32 else -1 for v in row])
  count=collections.Counter(v for row in ids for v in row if v>=0)
  occupancy.append(dict(m=m,pattern=pattern,timing_generation=0,cutoff=cutoff,local_routes=sum(count.values()),total_slots=m*8,empty_tokens=sum(all(v<0 for v in row) for row in ids),counts=[count[e] for e in range(32)],light_experts=sum(0<n<=cutoff for n in count.values()),heavy_experts=sum(n>cutoff for n in count.values()),route_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
rows=[]
for t in r['timings']:
 s=t['samples_us'];paired=[a/b for phase in (1,2) for a,b in zip(s['A'+str(phase)],s['B'+str(phase)])]
 halves={key:statistics.mean(values[:4])/statistics.mean(values[4:]) for key,values in s.items()}
 rows.append({**{k:t[k] for k in ('m','pattern','variant','cache_sweep','qualified','ratio','controls')},'mean_us':{key:statistics.mean(values) for key,values in s.items()},'median_us':t['medians_us'],'paired_A_over_B':paired,'mean_paired_ratio':statistics.mean(paired),'median_paired_ratio':statistics.median(paired),'first4_over_last4_means':halves,'any_half_drift_over10pct':any(not .9<=v<=1.1 for v in halves.values()),'raw_samples_us':s})
out={'status':'TIMING_SCOPE_AUDIT','source_report_sha256':hashlib.sha256((p/'composition_03/results.json').read_bytes()).hexdigest(),'occupancy':occupancy,'timings':rows,'limits':['Ratios are full local MoE helper+finalizer relative to scored helper, not stock SGLang or serving','Synthetic one-layer weight/routing fixture, single local rank, not rank-coherent maximum','Three targets timed sequentially split/fused/composed; separate A controls do not provide direct split-vs-composed ratio','Each cold/warm group first warms all four graphs for0.25s; no pertarget independent plateau detector','A/B pair order alternates, and two phases provide A/A and B/B drift checks','First4/last4 check is diagnostic only and does not retroactively revise original gate','Six original groups fail controls and remain held','Initial group may retain warmup/thermal bias; direct split-vs-composed followup should counterbalance target order and use perpair warm-state control']}
(p/'TIMING_OCCUPANCY_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'occupancy':occupancy,'firstcase_drift':[{k:x[k] for k in ('variant','cache_sweep','first4_over_last4_means','median_paired_ratio')} for x in rows if x['m']==768 and x['pattern']=='uniform'],'extra_half_drift_groups':sum(x['any_half_drift_over10pct'] for x in rows)}))

