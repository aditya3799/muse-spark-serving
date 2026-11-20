import json, pathlib, subprocess, sys, time
O=pathlib.Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-20/local_validation')
while True:
    state=json.loads((O/'matrix_state.json').read_text())
    if state.get('complete'): break
    time.sleep(10)
state={'scope':'Actual packaged CUDA factory, lifetime and fallback tests; virtual GPU peers, no physical TP8 transport.', 'jobs':[], 'complete':False}
for name in ('row_factory','output_n_factory'):
    item={'name':name,'start':time.time()};state['jobs'].append(item)
    (O/'factory_state.json').write_text(json.dumps(state,indent=2))
    print('START',name,flush=True)
    with (O/(name+'.log')).open('w') as log:
        try:
            p=subprocess.run([sys.executable,'-u',str(O/(name+'.py'))],stdout=log,stderr=subprocess.STDOUT,timeout=900)
            item['returncode']=p.returncode
        except subprocess.TimeoutExpired:item['timeout']=True
    item['end']=time.time();print('END',name,item.get('returncode','timeout'),flush=True)
    (O/'factory_state.json').write_text(json.dumps(state,indent=2))
state['complete']=True
(O/'factory_state.json').write_text(json.dumps(state,indent=2))
