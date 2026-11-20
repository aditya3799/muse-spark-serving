import json,os,pathlib,signal,subprocess,sys,time
R=pathlib.Path('/mnt/d/OpenKernelChallenge/serving');O=R/'research_2026-11-20/local_validation'
while True:
    p=O/'composition_state.json'
    if p.exists():
        try:
            if json.loads(p.read_text()).get('complete'):break
        except json.JSONDecodeError:pass
    time.sleep(10)
state={'complete':False,'scope':'Large-shape correctness completion and small synthetic-geometry sanitizer tests, no timing promotion.','jobs':[]}
def run(name,cmd,timeout):
    item={'name':name,'start':time.time(),'command':cmd};state['jobs'].append(item)
    (O/'post_state.json').write_text(json.dumps(state,indent=2));print('START',name,flush=True)
    with (O/(name+'.log')).open('w') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        item['pid']=process.pid
        try:item['returncode']=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            item['timeout']=True
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
    item['end']=time.time();(O/'post_state.json').write_text(json.dumps(state,indent=2));print('END',name,item.get('returncode','timeout'),flush=True)
    return item
for seed in (20112023,20112029):
    data=json.loads((O/f'row_chain_{seed}_16384.json').read_text())
    if not data.get('complete'):
        name=f'row_chain_completion_{seed}_16384'
        run(name,[sys.executable,'-u',str(R/'research_2026-11-19/prefill/real_row_chain.py'),'--seed',str(seed),'--rows','16384','--rounds','4','--out',str(O/(name+'.json'))],660)
san=json.loads((O/'tools/sanitizer_receipt.json').read_text())['binaries'][0]
for index,(mode,rows) in enumerate((('memcheck',8),('memcheck',512),('racecheck',512),('synccheck',512),('initcheck',8))):
    name=f'sanitizer_{mode}_{rows}'
    cmd=[san,'--tool',mode,'--error-exitcode','86']
    if mode=='memcheck':cmd+=['--padding','32','--destroy-on-device-error','context']
    cmd+=[sys.executable,'-u',str(O/'sanitizer_fixture.py'),'--rows',str(rows),'--seed',str(20112101+index)]
    outcome=run(name,cmd,600)
    if outcome.get('returncode')!=0:
        state['stopped_on_failure']=name
        break
state['complete']=True;(O/'post_state.json').write_text(json.dumps(state,indent=2))
