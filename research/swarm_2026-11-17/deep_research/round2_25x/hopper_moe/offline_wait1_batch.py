from pathlib import Path
import argparse,subprocess,sys,json,time
P=Path(__file__).resolve().parent;a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True);r={'status':'running','batches':[]};start=time.monotonic()
for name,script in [('wait1','offline_wait1_wrapper.py'),('wm48','offline_wm48_wrapper.py')]:
 cmd=[sys.executable,str(P/script),'--out',str(args.out/name)];done=subprocess.run(cmd);r['batches'].append({'name':name,'exit_code':done.returncode,'elapsed_total_seconds':time.monotonic()-start});(args.out/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['batches'][-1]),flush=True)
 if done.returncode:sys.exit(done.returncode)
r['status']='complete';(args.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
