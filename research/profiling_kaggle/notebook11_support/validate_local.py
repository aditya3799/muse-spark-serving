import hashlib,json,sys,zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from cache_study import CacheStudy
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest_11.json').read_text());helper=(root/'mapping_helper.py').read_text()
cpu=CacheStudy(manifest,helper,output_root=root,force_cpu=True,samples=5,repeats=1,trials=1)
cpu.environment()
for method in (cpu.lifecycle,cpu.input_contract,cpu.preservation_contract):assert method()['status']=='passed'
assert cpu.benchmark()['status']=='skipped';assert cpu.profile()['status']=='skipped';cpu.export()
path=root/'11_kaggle_cache_choice.ipynb'
nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:client.execute()
finally:
    nbformat.write(nb,root/'local_validation_11.executed.ipynb')
    if manager.has_kernel:manager.shutdown_kernel(now=True)
reports=sorted(root.glob('v4_cache_choice_*/report.json'),key=lambda p:p.stat().st_mtime)
r=json.loads(reports[-1].read_text())
for section in ('environment','lifecycle','input_contract','preservation_contract','benchmark'):assert r['sections'][section]['status']=='passed',r['sections'][section]
assert r['sections']['benchmark']['changed_input_output_checks']==2592
assert r['totals']=={'cases':48,'timing_rows':288,'raw_samples':3168,'comparisons':288}
with zipfile.ZipFile(reports[-1].parent.with_suffix('.zip')) as z:
    hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
result={'schema':'passed','cpu_gates':'passed','active_v4_modified':False,
    'statuses':{k:v['status'] for k,v in r['sections'].items()},'device':r['sections']['environment']['gpu_devices'],
    'preservation_contract':r['sections']['preservation_contract'],'benchmark':r['sections']['benchmark'],'totals':r['totals'],
    'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_path':str(reports[-1]),
    'note':'Static source prefixes only. Explicit refresh gate is separate from timings; no full integration or fused-kernel execution. GPU checks independent of profiler support.'}
(root/'LOCAL_VALIDATION_11.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
