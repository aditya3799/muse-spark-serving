import json, pathlib, hashlib, subprocess, math
ROOT=pathlib.Path(r'D:\OpenKernelChallenge\serving')
OUT=pathlib.Path(__file__).parent
cmd=['rg','-l','-uu','rungs',str(ROOT),'-g','*.json','-g','!sglang/**','-g','!references/**']
paths=subprocess.run(cmd,capture_output=True,text=True).stdout.splitlines()
jobs={}; rejected=[]
for ps in paths:
 p=pathlib.Path(ps)
 try: j=json.loads(p.read_text(encoding='utf-8-sig'))
 except Exception as e: rejected.append([ps,type(e).__name__]);continue
 if not isinstance(j,dict):continue
 r=j.get('result',{})
 if not isinstance(r,dict) or not isinstance(r.get('rungs'),list):continue
 id=j.get('id',str(p))
 if id not in jobs:jobs[id]={'id':id,'source_paths':[],'result':{k:v for k,v in r.items() if k not in ('startup_log','sandbox')},'root_status':j.get('status'),'mode':j.get('mode')}
 jobs[id]['source_paths'].append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 log=r.get('startup_log',{})
 if log:jobs[id]['startup_log_scope']={k:v for k,v in log.items() if k!='text'}
for j in jobs.values():
 r=j['result'];f=r.get('frontier',{});points=f.get('points',[])
 if points:j['geomean_from_reported_ratios']=math.prod(p['ratio'] for p in points)**(1/len(points))
 j['exposed_component_timers']=[k for k in r if any(s in k for s in ['draft','target','decode_time','prefill_time','request'])]
report={'scope':'Saved local serving job summaries only; no remote requests or GPU calls','jobs':list(jobs.values()),'matching_json_files':len(paths),'parse_failures':rejected,'missing_evidence':['Per-request prompt/output lengths, durations, TTFT and cache events','Per-repetition raw paired observations','Draft versus target/verify elapsed breakdown','Serving operator or queue critical-path trace','Official frontier interpolation/token-accounting implementation']}
(OUT/'SAVED_BENCHMARK_INVENTORY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'matching_json_files':len(paths),'unique_jobs':len(jobs),'jobs':[{'id':j['id'],'score':j['result'].get('frontier',{}).get('score'),'passed':j['result'].get('passed'),'status':j['root_status'],'rungs':j['result'].get('rungs'),'shape':j['result'].get('shape')} for j in jobs.values()]},indent=2))


