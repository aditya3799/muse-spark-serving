import json,os,pathlib,subprocess,sys,time
R=pathlib.Path('/mnt/d/OpenKernelChallenge/serving');O=R/'research_2026-11-20/local_validation'
jobs=[]
for seed in (20112013,20112017):
 for rows in (8,32,128,256,512,768,3072):
  jobs.append((f'adapter_{seed}_{rows}',[str(O/'adapter_probe.py'),'--seed',str(seed),'--rows',str(rows)]))
for seed in (20112023,20112029):
 for rows in (512,768,3072,8192,16384):
  name=f'row_chain_{seed}_{rows}'
  jobs.append((name,[str(R/'research_2026-11-19/prefill/real_row_chain.py'),'--seed',str(seed),'--rows',str(rows),'--rounds','12','--out',str(O/(name+'.json'))]))
 for b in (1,4,16,32):
  jobs.append((f'decode_chain_{seed}_{b}',[str(R/'research_2026-11-19/decode/output_shard_coupled.py'),'--seed',str(seed+b),'--batches',str(b),'--rounds','12','--risks','partial','full']))
state={'scope':'Local RTX3050 actual adapter and real-weight component chains. No target full model, physical TP8 NCCL, or H200 score. Every GPU child runs serially.','started':time.time(),'jobs':[],'complete':False}
for name,args in jobs:
 item={'name':name,'start':time.time()};state['jobs'].append(item)
 (O/'matrix_state.json').write_text(json.dumps(state,indent=2));print('START',name,flush=True)
 with (O/(name+'.log')).open('w') as log:
  try:
   p=subprocess.run([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT,timeout=420)
   item['returncode']=p.returncode
  except subprocess.TimeoutExpired:item['timeout']=True
 item['end']=time.time();(O/'matrix_state.json').write_text(json.dumps(state,indent=2));print('END',name,item.get('returncode','timeout'),flush=True)
state['complete']=True;state['end']=time.time();(O/'matrix_state.json').write_text(json.dumps(state,indent=2))
