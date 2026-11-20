import hashlib,json,sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from replay_study import ReplayStudy

root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest_06.json').read_text());helper=(root/'mapping_helper.py').read_text()
cpu=ReplayStudy(manifest,helper,output_root=root,force_cpu=True,samples=5,repeats=1,trials=1)
cpu.environment();assert cpu.correctness()['status']=='passed'
assert cpu.benchmark()['status']=='skipped';assert cpu.profile()['status']=='skipped';cpu.export()
path=root/'06_kaggle_replay_schedule.ipynb'
if not path.exists():path=root.parent/path.name
nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:client.execute()
finally:
    nbformat.write(nb,root/'local_validation_06.executed.ipynb')
    if manager.has_kernel:manager.shutdown_kernel(now=True)
reports=sorted(root.glob('v4_replay_schedule_*/report.json'),key=lambda p:p.stat().st_mtime)
r=json.loads(reports[-1].read_text())
for section in ('environment','correctness','benchmark'):assert r['sections'][section]['status']=='passed',r['sections'][section]
assert r['sections']['correctness']['input_cases']==192
assert r['sections']['benchmark']['changed_input_schedule_checks']==360
assert r['totals']=={'cases':30,'timing_rows':180,'raw_samples':1980}
for name,h in json.loads((reports[-1].parent/'checksums.json').read_text()).items():assert hashlib.sha256((reports[-1].parent/name).read_bytes()).hexdigest()==h
result={'schema':'passed','cpu_correctness_and_export':'passed','statuses':{k:v['status'] for k,v in r['sections'].items()},
        'device':r['sections']['environment']['gpu_devices'],'correctness':r['sections']['correctness'],'benchmark':r['sections']['benchmark'],'totals':r['totals'],
        'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_path':str(reports[-1]),
        'note':'Local inference-mode source-slice validation only; no full model, real alignment/GEMM or H200 execution.'}
(root/'LOCAL_VALIDATION_06.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
