from pathlib import Path
import argparse,subprocess,sys,json,hashlib,time
P=Path(__file__).resolve().parent;D=P/'sm90_corrected'
a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();args.out.mkdir(exist_ok=True,parents=True)
r={'status':'running','scope':'12 offline SM90 builds only; no CUDA context/module load or runtime proof','batches':[]};start=time.monotonic()
for name,script,count in [('virtual','offline_virtual_k_sm90.py',8),('fused','offline_w13_fused_sm90.py',4)]:
 dest=args.out/name;dest.mkdir(exist_ok=True);cmd=[sys.executable,str(D/script),'--out',str(dest),'--count',str(count)]
 completed=subprocess.run(cmd);item={'name':name,'exit_code':completed.returncode,'elapsed_total_seconds':time.monotonic()-start};r['batches'].append(item)
 if (dest/'results.json').exists():item['receipt_sha256']=hashlib.sha256((dest/'results.json').read_bytes()).hexdigest()
 (args.out/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(item),flush=True)
 if completed.returncode:sys.exit(completed.returncode)
r['status']='complete';(args.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
