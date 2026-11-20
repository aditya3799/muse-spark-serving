from pathlib import Path
import json,subprocess,hashlib,shutil,re
p=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-17/deep_research/round3_structural/aux_speculation');out=p/'packed_materialize_resource_audit';out.mkdir(exist_ok=True)
exe=Path('/home/aditya/.venvs/okc-marlin-tiles/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump')
reports=[]
for i,meta in enumerate(sorted(Path('/home/aditya/.triton/cache').rglob('packed_dequant_kernel.json'))):
 d=meta.parent;row={'cache':str(d),'metadata':json.loads(meta.read_text())};dest=out/str(i);dest.mkdir(exist_ok=True)
 for name in ('packed_dequant_kernel.json','packed_dequant_kernel.ptx','packed_dequant_kernel.cubin','packed_dequant_kernel.ttgir'):
  f=d/name
  if f.exists():shutil.copyfile(f,dest/name)
 cubin=d/'packed_dequant_kernel.cubin';r=subprocess.run([str(exe),'--dump-resource-usage',str(cubin)],capture_output=True,text=True,timeout=20);row.update(resource_exit=r.returncode,resources=r.stdout,stderr=r.stderr,cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest());(dest/'resources.txt').write_text(r.stdout+r.stderr)
 sass=subprocess.run([str(exe),'--dump-sass',str(cubin)],capture_output=True,text=True,timeout=20);(dest/'sass.txt').write_text(sass.stdout);row['sass_local_loads']=len(re.findall(r'\bLDL(?:\.|\s)',sass.stdout));row['sass_local_stores']=len(re.findall(r'\bSTL(?:\.|\s)',sass.stdout));ptx=(d/'packed_dequant_kernel.ptx').read_text();row['ptx_local_loads']=len(re.findall(r'ld\.local',ptx));row['ptx_local_stores']=len(re.findall(r'st\.local',ptx));row['ptx_barriers']=len(re.findall(r'bar\.sync',ptx));row['ptx_shared_loads']=len(re.findall(r'ld\.shared',ptx));row['ptx_shared_stores']=len(re.findall(r'st\.shared',ptx));reports.append(row)
(out/'results.json').write_text(json.dumps({'status':'OFFLINE_RESOURCE_AUDIT','cuda_context_created':False,'cases':reports},indent=2)+'\n')
for row in reports:print(row['resources'],{k:v for k,v in row.items() if k.startswith(('ptx_','sass_'))})
