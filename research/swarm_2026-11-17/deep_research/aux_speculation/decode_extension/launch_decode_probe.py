import argparse,hashlib,json,os,subprocess,sys,time
import importlib.metadata as metadata
from pathlib import Path
P=Path(__file__).resolve().parent;A=P.parent;S=A.parents[2]/'sglang/python/sglang'
pa=argparse.ArgumentParser();pa.add_argument('--name',default='direct_graph_01');pa.add_argument('--seconds',type=int,default=60);pa.add_argument('--script',default='probe_decode_graph.py');pa.add_argument('--fused',action='store_true');pa.add_argument('--hot',action='store_true');pa.add_argument('--hidden',type=int,default=64);a=pa.parse_args()
if a.fused:
    compiler=metadata.distribution('nvidia-cuda-nvcc');nvcc=next(Path(compiler.locate_file(f)).resolve() for f in compiler.files if str(f).endswith('/bin/nvcc'));cuda=nvcc.parent.parent
    os.environ.update(CUDA_HOME=str(cuda),MAX_JOBS='1',SGLANG_JIT_LOG_RESOURCE_USAGE='1')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(cuda/'bin')+os.pathsep+os.environ.get('PATH','')
    os.environ['LD_LIBRARY_PATH']=str(cuda/'lib64')+':/usr/lib/wsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')
    headers=Path(sys.prefix)/'python-headers/usr/include';os.environ['CPATH']=str(headers/'python3.12')+':'+str(headers)+':'+os.environ.get('CPATH','')
out=P/a.name;out.mkdir(exist_ok=True);snap=out/'source_snapshots';snap.mkdir(exist_ok=True)
files=list(P.glob('*.py'))+list((P/(('hot_' if a.hot else '')+('fused_sources' if a.fused else 'direct_sources'))).rglob('*.py'))+list((P/(('hot_' if a.hot else '')+('fused_sources' if a.fused else 'direct_sources'))).rglob('*.cuh'))+[S/'srt/model_executor/input_buffers.py',S/'srt/model_executor/runner_utils/buffers.py',S/'srt/model_executor/runner_backend/full_cuda_graph_backend.py',S/'srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py',S/'srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py']
files += [A/'base_sources/python/sglang/srt/models/muse_spark.py',P/'direct_sources/python/sglang/srt/layers/decode_aux_output.py',A/'fusion_store/norm_residual_aux.py',A/'fusion_store/norm_residual_aux.cuh',A/'fusion_store/runtime_imports.py',S/'kernels/ops/layernorm/muse_spark_norm_residual_jit.py',S/'kernels/jit/csrc/muse_spark/norm_residual.cuh']
sources={}
for path in files:
    data=path.read_bytes();name='%03d_%s'%(len(sources),path.name);(snap/name).write_bytes(data);sources[str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'snapshot':name}
cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(a.seconds)+'s',sys.executable,str(P/a.script),'--out',str(out/'results.json'),'--hidden',str(a.hidden)]+(['--fused'] if a.fused else [])+(['--hot'] if a.hot else [])
record={'start':time.time(),'status':'running','sources':sources,'command':cmd}
def save():(out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
save()
with (out/'telemetry.csv').open('w') as te:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu','--format=csv','-l','1'],stdout=te,stderr=subprocess.DEVNULL)
    try:
        with (out/'stdout.log').open('w') as so,(out/'stderr.log').open('w') as se:
            process=subprocess.Popen(cmd,stdout=so,stderr=se,env=dict(os.environ,PYTHONUNBUFFERED='1'));record['pid']=process.pid;save();rc=process.wait()
        record.update(status='exited',exit_code=rc,finish=time.time());save()
    finally:monitor.terminate();monitor.wait(timeout=5)
print(json.dumps({k:v for k,v in record.items() if k!='sources'}));raise SystemExit(rc)
