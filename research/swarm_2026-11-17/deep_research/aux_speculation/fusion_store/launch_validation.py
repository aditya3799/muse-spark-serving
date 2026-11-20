import hashlib,importlib.metadata as metadata,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;A=P.parent;out=P/'fusion_validation_04';out.mkdir(exist_ok=True)
compiler=metadata.distribution('nvidia-cuda-nvcc');nvcc=next(Path(compiler.locate_file(f)).resolve() for f in compiler.files if str(f).endswith('/bin/nvcc'));cuda=nvcc.parent.parent
os.environ.update(CUDA_HOME=str(cuda),MAX_JOBS='1',PYTHONUNBUFFERED='1',SGLANG_JIT_LOG_RESOURCE_USAGE='1')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(cuda/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['LD_LIBRARY_PATH']=str(cuda/'lib64')+':/usr/lib/wsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')
headers=Path(sys.prefix)/'python-headers/usr/include';os.environ['CPATH']=str(headers/'python3.12')+':'+str(headers)+':'+os.environ.get('CPATH','')
snap=out/'source_snapshots';snap.mkdir(exist_ok=True);sources={}
for path in list(P.glob('*.py'))+list(P.glob('*.cuh'))+list((P/'integration_sources').rglob('*.py'))+list((A/'candidate_sources').rglob('*.py'))+[A/'prefill_aux_output.py',A/'test_integrated_methods.py']:
    data=path.read_bytes();name='%03d_%s'%(len(sources),path.name);(snap/name).write_bytes(data);sources[str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'snapshot':name}
record={'start':time.time(),'status':'running','sources':sources,'nvcc':str(nvcc),'runs':[]}
def save():(out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
save();deadline=time.monotonic()+110;code=0
jobs=[('owner',[str(A/'test_integrated_methods.py'),'--device','cuda','--out',str(out/'owner.json')]),('primitive',[str(P/'probe_fusion.py'),'--out',str(out/'primitive.json')]),('integrated_graph',[str(P/'probe_integrated_graph.py'),'--out',str(out/'integrated_graph.json')]),('timing',[str(P/'bench_fusion.py'),'--out',str(out/'timing.json')])]
with (out/'telemetry.csv').open('w') as te:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu','--format=csv','-l','1'],stdout=te,stderr=subprocess.DEVNULL)
    try:
        for name,args in jobs:
            remaining=max(1,int(deadline-time.monotonic()));cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(remaining)+'s',sys.executable,*args]
            run={'name':name,'start':time.time(),'command':cmd};record['runs'].append(run);save()
            with (out/(name+'_stdout.log')).open('w') as so,(out/(name+'_stderr.log')).open('w') as se:
                process=subprocess.Popen(cmd,stdout=so,stderr=se);run['pid']=process.pid;save();code=process.wait()
            run.update(exit_code=code,finish=time.time());save();print(json.dumps(run),flush=True)
            if code:break
    finally:
        monitor.terminate();monitor.wait(timeout=5);record.update(status='exited',exit_code=code,finish=time.time());save()
raise SystemExit(code)
