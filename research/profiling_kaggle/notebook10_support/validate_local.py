import hashlib,json,sys,zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from contract_study import ContractStudy
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest_10.json').read_text());helper=(root/'mapping_helper.py').read_text()
cpu=ContractStudy(manifest,helper,output_root=root,force_cpu=True,samples=5,repeats=1,trials=1)
cpu.environment();assert cpu.lifecycle()['status']=='passed';assert cpu.input_contract()['status']=='passed'
assert cpu.benchmark()['status']=='skipped';assert cpu.profile()['status']=='skipped';cpu.export()
path=root/'10_kaggle_router_contract.ipynb'
nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:client.execute()
finally:
    nbformat.write(nb,root/'local_validation_10.executed.ipynb')
    if manager.has_kernel:manager.shutdown_kernel(now=True)
reports=sorted(root.glob('v4_router_contract_*/report.json'),key=lambda p:p.stat().st_mtime)
r=json.loads(reports[-1].read_text())
for section in ('environment','lifecycle','input_contract','benchmark'):assert r['sections'][section]['status']=='passed',r['sections'][section]
assert r['sections']['input_contract']['input_cases']==40 and r['sections']['input_contract']['output_checks']==80
assert r['sections']['benchmark']['changed_input_output_checks']==1296
assert r['totals']=={'cases':18,'timing_rows':144,'raw_samples':1584,'comparisons':252}
with zipfile.ZipFile(reports[-1].parent.with_suffix('.zip')) as z:
    hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
result={'schema':'passed','cpu_lifecycle_and_input_contract':'passed','active_v4_modified':False,
    'statuses':{k:v['status'] for k,v in r['sections'].items()},'device':r['sections']['environment']['gpu_devices'],
    'lifecycle':r['sections']['lifecycle'],'input_contract':r['sections']['input_contract'],'benchmark':r['sections']['benchmark'],'totals':r['totals'],
    'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_path':str(reports[-1]),
    'note':'Synthetic dtype contracts and source prefixes only. Actual router, full model and startup not executed. GPU timing/correctness checked independently from profiler support.'}
(root/'LOCAL_VALIDATION_10.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
