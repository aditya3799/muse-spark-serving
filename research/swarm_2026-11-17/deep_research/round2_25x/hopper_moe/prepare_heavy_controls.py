from pathlib import Path
import json,ast
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');D=P/'sm90_heavy_controls';D.mkdir(exist_ok=True);base=P/'sm90_corrected';jobs=json.loads((base/'w13_fused_jobs.json').read_text())['jobs'];jobs=[j for j in jobs if j['policy']=='heavy']
for j in jobs:j['job_id']+='_canonical_control';j['policy']='canonical_control'
(D/'w13_fused_jobs.json').write_text(json.dumps({'scope':'Exact unchanged canonical control for attribution of fusion spills','jobs':jobs},indent=2)+'\n',encoding='utf-8')
s=(base/'offline_w13_fused_sm90.py').read_text(encoding='utf-8-sig').replace('generated=render(k.code)','generated=k.code').replace('default=4','default=2');(D/'offline_w13_fused_sm90.py').write_text(s,encoding='utf-8');ast.parse(s)
wrapper='''from pathlib import Path
import argparse,subprocess,sys
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args()
sys.exit(subprocess.run([sys.executable,str(P/'sm90_heavy_controls/offline_w13_fused_sm90.py'),'--out',args.out,'--count','2']).returncode)
'''
(P/'offline_heavy_controls.py').write_text(wrapper,encoding='utf-8')
