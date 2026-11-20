from pathlib import Path
import json,copy,hashlib
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');D=P/'sm90_heavy_cap';D.mkdir(exist_ok=True);src=P.parent/'expert_partition/SM90_HEAVY_CAP_COMPILE_PROPOSAL.json';jobs=[]
for j in json.loads(src.read_text())['jobs']:
 if j['tuning_config']['block_shape'][0] not in (128,144):continue
 c={**j['meta'],**j['compute_config'],**j['tuning_config'],'use_cp_async':True}
 for mode in ('canonical','fused'):jobs.append({'stage':'w13','policy':mode,'job_id':j['name']+'_'+mode,'configuration':copy.deepcopy(c),'baseline':copy.deepcopy(c),'intervals':[]})
(D/'w13_fused_jobs.json').write_text(json.dumps({'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'scope':'Resource-only; native opcode width differs from scored heavy policy','jobs':jobs},indent=2)+'\n',encoding='utf-8')
s=(P/'sm90_corrected/offline_w13_fused_sm90.py').read_text().replace('generated=render(k.code)',"generated=render(k.code) if policy=='fused' else k.code");(D/'offline_w13_fused_sm90.py').write_text(s,encoding='utf-8')
wrapper="from pathlib import Path\nimport argparse,subprocess,sys\nP=Path(__file__).resolve().parent\na=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args()\nsys.exit(subprocess.run([sys.executable,str(P/'sm90_heavy_cap/offline_w13_fused_sm90.py'),'--out',args.out,'--count','4']).returncode)\n";(P/'offline_heavy_cap.py').write_text(wrapper,encoding='utf-8')
