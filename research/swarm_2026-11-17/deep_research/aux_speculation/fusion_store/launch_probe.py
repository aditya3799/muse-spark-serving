import argparse,hashlib,json,os,subprocess,sys,time
import importlib.metadata as metadata
from pathlib import Path
P=Path(__file__).resolve().parent
compiler=metadata.distribution('nvidia-cuda-nvcc')
nvcc=next(Path(compiler.locate_file(f)).resolve() for f in compiler.files if str(f).endswith('/bin/nvcc'));cuda=nvcc.parent.parent
os.environ.update(CUDA_HOME=str(cuda),MAX_JOBS='1')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(cuda/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['LD_LIBRARY_PATH']=str(cuda/'lib64')+':/usr/lib/wsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')
headers=Path(sys.prefix)/'python-headers/usr/include'
os.environ['CPATH']=str(headers/'python3.12')+':'+str(headers)+':'+os.environ.get('CPATH','')
pa=argparse.ArgumentParser();pa.add_argument('--name',default='primitive_gpu_01');pa.add_argument('--seconds',type=int,default=160);pa.add_argument('--script',default='probe_fusion.py');a=pa.parse_args()
out=P/a.name;out.mkdir(exist_ok=True);snap=out/'source_snapshots';snap.mkdir(exist_ok=True)
source={}
for path in list(P.glob('*.py'))+list(P.glob('*.cuh'))+[P.parents[3]/'sglang/python/sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py',P.parents[3]/'sglang/python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh']:
    data=path.read_bytes();name='%02d_%s'%(len(source),path.name);(snap/name).write_bytes(data);source[str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'snapshot':name}
cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(a.seconds)+'s',sys.executable,str(P/a.script),'--out',str(out/'results.json')]
record={'start':time.time(),'status':'running','command':cmd,'sources':source,'nvcc':str(nvcc)}
def save(): (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
save()
with (out/'telemetry.csv').open('w') as telemetry:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu','--format=csv','-l','1'],stdout=telemetry,stderr=subprocess.DEVNULL)
    try:
        with (out/'stdout.log').open('w') as so,(out/'stderr.log').open('w') as se:
            process=subprocess.Popen(cmd,stdout=so,stderr=se,env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),MAX_JOBS='1',PYTHONUNBUFFERED='1',SGLANG_JIT_LOG_RESOURCE_USAGE='1'))
            record['pid']=process.pid;save();rc=process.wait()
        record.update(status='exited',exit_code=rc,finish=time.time());save()
    finally:monitor.terminate();monitor.wait(timeout=5)
print(json.dumps({k:v for k,v in record.items() if k!='sources'}));raise SystemExit(rc)
