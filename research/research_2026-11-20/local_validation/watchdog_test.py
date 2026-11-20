import ast,json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-20/local_validation')
node=next(n for n in ast.parse((root/'post_suite.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
O=root/'watchdog_test';O.mkdir(exist_ok=True);state={'jobs':[]}
ns=dict(globals());exec(compile(ast.Module(body=[node],type_ignores=[]),'post_suite.run','exec'),ns)
receipt=O/'grandchild_pid.txt'
code="import pathlib,subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
result=ns['run']('owned_cpu_timeout',[sys.executable,'-c',code,str(receipt)],2)
pid=int(receipt.read_text());time.sleep(.2)
stat=pathlib.Path(f'/proc/{pid}/stat')
process_state=stat.read_text().split(') ',1)[1].split()[0] if stat.exists() else 'gone'
assert result.get('timeout') and process_state in ('gone','Z'),(result,process_state)
report={'complete':True,'correct':True,'scope':'Actual supervisor timeout function tested on a CPU-only child that spawned a grandchild; no GPU fault injected.','timeout_detected':True,'grandchild_state':process_state}
(root/'watchdog_test.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
