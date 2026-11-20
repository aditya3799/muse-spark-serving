from pathlib import Path
import json,copy,ast
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');S=P/'sm90_corrected';D=P/'sm90_wm48';D.mkdir(exist_ok=True)
for f in list(S.glob('virtual_k_*.cuh'))+[S/'virtual_k_codegen.py']:(D/f.name).write_bytes(f.read_bytes())
j=copy.deepcopy(next(j for j in json.loads((S/'virtual_k_jobs.json').read_text())['jobs'] if j['job_id']=='w13_m32_actual_virtual_m2'));j['job_id']='w13_m32_bm48_wm48_virtual';j['configuration']['warp_shape'][0]=48;j['baseline']=copy.deepcopy(j['configuration']);j['policy']='virtual'
(D/'virtual_k_jobs.json').write_text(json.dumps({'scope':'Native opcode width changes N24 toN48. Same K16 primitive size and explicit K banks/tree; no bitwise equivalence claim.','jobs':[j]},indent=2)+'\n',encoding='utf-8')
s=(S/'offline_virtual_k_sm90.py').read_text().replace('default=8','default=1');(D/'offline_virtual_k_sm90.py').write_text(s,encoding='utf-8')
s=(S/'audit_writer_ownership.py').read_text();(D/'audit_writer_ownership.py').write_text(s,encoding='utf-8')
W=P/'wait1';s=(S/'offline_virtual_k_sm90.py').read_text().replace('from virtual_k_codegen import render_virtual_k','from wait1_codegen import render_wait1').replace("P/'virtual_k_codegen.py'","P/'wait1_codegen.py'").replace("P.glob('virtual_k_*.cuh')","P.glob('wait1_*.cuh')").replace("P/'virtual_k_jobs.json'","P/'jobs.json'").replace('render_virtual_k(k.code','render_wait1(k.code').replace("enabled=policy.startswith('virtual')","enabled=policy=='wait1'");(W/'offline_wait1_sm90.py').write_text(s,encoding='utf-8');ast.parse(s)
for label,path,count in [('wm48','sm90_wm48/offline_virtual_k_sm90.py',1),('wait1','wait1/offline_wait1_sm90.py',8)]:
 s="from pathlib import Path\nimport argparse,subprocess,sys\nP=Path(__file__).resolve().parent\na=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args()\nsys.exit(subprocess.run([sys.executable,str(P/"+repr(path)+"),'--out',args.out,'--count',"+repr(str(count))+"]).returncode)\n";(P/('offline_'+label+'_wrapper.py')).write_text(s,encoding='utf-8')
print('Prepared WM48 and wait1 wrappers; CPU only')
