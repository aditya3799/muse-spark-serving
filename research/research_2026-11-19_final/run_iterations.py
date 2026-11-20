import argparse, fcntl, importlib.util, json, os, pathlib, resource, subprocess, sys, time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
ROOT=pathlib.Path('/mnt/d/OpenKernelChallenge/serving')
OUT=ROOT/'research_2026-11-19_final'
PY=sys.executable

def load(path,name):
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def child(kind,seed):
    import torch
    torch.set_num_threads(1)
    if kind=='row':
        m=load(ROOT/'research_2026-11-19/prefill/real_row_chain.py','real_row_final')
        sys.argv=[str(m.__file__),'--seed',str(seed),'--out',str(OUT/f'row_{seed}.json'),
                  '--rows','512','768','3072','8192','16384','--rounds','8']
        with torch.inference_mode():m.main()
    elif kind=='output_n':
        m=load(ROOT/'research_2026-11-19/decode/output_shard_coupled.py','n_final')
        d=OUT/f'output_n_{seed}';d.mkdir()
        a=argparse.Namespace(seed=seed,batches=[1,4,16,32],rounds=12,risks=['partial','full'],frozen=None,trace=False)
        with torch.inference_mode():m.main(a,d)
    elif kind=='selector':
        source=ROOT/'research_2026-11-19_round2/coordination/target_argmax_postops'
        d=OUT/f'selector_{seed}';d.mkdir()
        for filename in ('target_select.py','timing_device_guard.py'):
            text=(source/filename).read_text().replace('438937',str(seed))
            (d/filename).write_text(text)
        m=load(d/'timing_device_guard.py','selector_final');m.main()

def main():
    OUT.mkdir(exist_ok=True)
    jobs=[('row',19011901),('output_n',19011903),('selector',19011907),
          ('row',19011911),('output_n',19011913)]
    state={'scope':'Local component and wider-chain diagnostics; no H200 simulation or full-serving execution',
           'started':time.time(),'jobs':[],'complete':False}
    with (ROOT/'research_2026-11-18_rl/.gpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for kind,seed in jobs:
            log=OUT/f'{kind}_{seed}.log';r={'kind':kind,'seed':seed,'start':time.time(),'log':str(log)}
            state['jobs'].append(r);(OUT/'RUN_STATE.json').write_text(json.dumps(state,indent=2))
            print('START',kind,seed,flush=True)
            with log.open('w') as f:
                try:
                    p=subprocess.run([PY,__file__,'--child',kind,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
                    r['returncode']=p.returncode
                except subprocess.TimeoutExpired:r['timeout']=True
            r['end']=time.time();(OUT/'RUN_STATE.json').write_text(json.dumps(state,indent=2))
            print('DONE',kind,seed,r.get('returncode','timeout'),flush=True)
    state['complete']=True;state['finished']=time.time();(OUT/'RUN_STATE.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--seed',type=int);a=p.parse_args()
    child(a.child,a.seed) if a.child else main()
