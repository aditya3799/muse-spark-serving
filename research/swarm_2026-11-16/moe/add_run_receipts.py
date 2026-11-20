from pathlib import Path
import hashlib,json,datetime,ast
p=Path(__file__).resolve().parent
known='f6888492daceb7c331f569af14a865e2f8a3e3d3e028f0ae51f85dc66ac6424a'
assert hashlib.sha256((p/'frozen_helper_probe.py').read_bytes()).hexdigest()==known
prefix='/mnt/d/OpenKernelChallenge/serving';python='/home/aditya/.venvs/okc-marlin-tiles/bin/python'
for name,e,m,session,exit_code,bounded in [('frozen_recovered_m1',4,1,20569,0,False),('frozen_recovered_m8',4,8,2556,0,False),('frozen_e32_m33',32,33,93434,0,True),('frozen_e32_m128',32,128,30405,None,True)]:
 argv=['wsl','-d','Ubuntu','--']
 if bounded:argv+=['/usr/bin/timeout','--signal=TERM','--kill-after=15s','480s']
 argv+=[python,prefix+'/swarm_2026-11-16/moe/launch_frozen_probe.py','--sglang-root',prefix+'/sglang','--out',prefix+'/swarm_2026-11-16/moe/'+name,'--hidden','4096','--experts',str(e),'--tokens',str(m)]
 receipt={'scope':'Agent run receipt from exact exec tool command and observed process outcome; original results.json preserved','argv':argv,'geometry':{'local_experts':e,'global_experts':e*8,'hidden':4096,'intermediate':4096,'top_k':8,'tokens':[m]},'session':session,'exit_code':exit_code,'status':'completed' if exit_code==0 else 'running','executed_harness_sha256':known,'launcher_sha256':hashlib.sha256((p/'launch_frozen_probe.py').read_bytes()).hexdigest(),'source_audit':'HARNESS_AUDIT.json recorded same harness hash before runs','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 out=p/name;out.mkdir(exist_ok=True);(out/'run_receipt.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
f=p/'memory_probe.py';s=f.read_text(encoding='utf-8')
s=s.replace("report={'mode':a.mode,", "report={'mode':a.mode,'status':'running','argv':sys.argv,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'geometry':{'local_experts':e,'global_experts':e*8,'hidden':h,'intermediate':i,'top_k':8,'tokens':a.tokens},'measurement_scope':'PyTorch allocator allocated/reserved only; external CUDA driver/module graph memory not measured',")
s=s.replace("report['completed']=len(report['graphs'])==len(a.tokens);", "report['completed']=len(report['graphs'])==len(a.tokens);report['status']='completed' if report['completed'] else 'stopped_cooperatively';")
ast.parse(s);f.write_bytes(s.encode('utf-8'))
print('run receipts saved; future memory report metadata corrected')
