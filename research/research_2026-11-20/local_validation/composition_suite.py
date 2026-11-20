import json,pathlib,subprocess,sys,time
O=pathlib.Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-20/local_validation')
while True:
    p=O/'factory_state.json'
    if p.exists():
        try:
            if json.loads(p.read_text()).get('complete'):break
        except json.JSONDecodeError:pass
    time.sleep(10)
state={'complete':False,'jobs':[]}
for m in (8,256,512,3072):
    job={'rows':m,'start':time.time()};state['jobs'].append(job)
    (O/'composition_state.json').write_text(json.dumps(state,indent=2));print('START composed',m,flush=True)
    with (O/f'exported_chain_{m}.log').open('w') as log:
        try:
            p=subprocess.run([sys.executable,'-u',str(O/'exported_chain.py'),'--rows',str(m)],stdout=log,stderr=subprocess.STDOUT,timeout=420)
            job['returncode']=p.returncode
        except subprocess.TimeoutExpired:job['timeout']=True
    job['end']=time.time();print('END composed',m,job.get('returncode','timeout'),flush=True)
    (O/'composition_state.json').write_text(json.dumps(state,indent=2))
state['complete']=True;(O/'composition_state.json').write_text(json.dumps(state,indent=2))
