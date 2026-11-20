import argparse,os,pathlib,subprocess,sys,time,json,hashlib,importlib.metadata,shutil,threading
p=argparse.ArgumentParser();p.add_argument('--script',required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=int,default=180);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
script=pathlib.Path(a.script);snapshot=out/'source_snapshot.py';shutil.copyfile(script,snapshot)
dist=importlib.metadata.distribution('nvidia-cuda-nvcc');nvcc=next(pathlib.Path(dist.locate_file(f)).resolve() for f in dist.files if str(f).endswith('/bin/nvcc'));cuda=nvcc.parent.parent
env=os.environ.copy();env.update(MAX_JOBS='1',HUMMING_DISABLE_PARALLEL_BUILD='1',CUDA_HOME=str(cuda),PATH=str(pathlib.Path(sys.executable).parent)+':'+str(cuda/'bin')+':'+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+':/usr/lib/wsl/lib:'+env.get('LD_LIBRARY_PATH',''),CPATH=sys.prefix+'/python-headers/usr/include/python3.12:'+str(cuda/'include')+':'+env.get('CPATH',''))
cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(a.seconds)+'s',sys.executable,str(script),'--out',str(out)]+(a.args[1:] if a.args[:1]==['--'] else a.args)
receipt={'command':cmd,'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'started_unix_seconds':time.time(),'status':'RUNNING','scope':'local isolated research, no submissions'}
(out/'run_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
stop=threading.Event()
def telemetry():
 with (out/'loaded_telemetry.jsonl').open('w') as f:
  while not stop.is_set():
   try:
    sample=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
    f.write(json.dumps({'unix_seconds':time.time(),'exit_code':sample.returncode,'value':sample.stdout.strip()})+'\n');f.flush()
   except Exception as error:f.write(json.dumps({'error':repr(error)})+'\n');f.flush()
   stop.wait(1)
thread=threading.Thread(target=telemetry,daemon=True);thread.start()
with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:r=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr)
stop.set();thread.join(timeout=4)
receipt.update(status='EXITED',exit_code=r.returncode,finished_unix_seconds=time.time());(out/'run_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));print((out/'stdout.log').read_text()[-2000:]);print((out/'stderr.log').read_text()[-2000:]);sys.exit(r.returncode)

