import hashlib,json,sys,zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from graph_study import GraphStudy
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest_09.json').read_text());helper=(root/'mapping_helper.py').read_text()
cpu=GraphStudy(manifest,helper,output_root=root,force_cpu=True,samples=5,repeats=1,trials=1)
cpu.environment();assert cpu.lifecycle()['status']=='passed'
assert cpu.benchmark()['status']=='skipped';assert cpu.profile()['status']=='skipped';cpu.export()
path=root/'09_kaggle_graph_boundaries.ipynb'
nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:client.execute()
finally:
    nbformat.write(nb,root/'local_validation_09.executed.ipynb')
    if manager.has_kernel:manager.shutdown_kernel(now=True)
reports=sorted(root.glob('v4_graph_boundaries_*/report.json'),key=lambda p:p.stat().st_mtime)
r=json.loads(reports[-1].read_text())
for section in ('environment','lifecycle','benchmark'):assert r['sections'][section]['status']=='passed',r['sections'][section]
assert r['sections']['benchmark']['changed_input_output_checks']==4428
assert r['totals']=={'cases':27,'timing_rows':162,'raw_samples':1782,'comparisons':135}
with zipfile.ZipFile(reports[-1].parent.with_suffix('.zip')) as z:
    hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
result={'schema':'passed','cpu_lifecycle':'passed','active_v4_modified':False,
    'statuses':{k:v['status'] for k,v in r['sections'].items()},'device':r['sections']['environment']['gpu_devices'],
    'lifecycle':r['sections']['lifecycle'],'benchmark':r['sections']['benchmark'],'totals':r['totals'],
    'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_path':str(reports[-1]),
    'note':'Independent source-prefix stages only. GPU timing/correctness checked separately from profiler support; actual model and startup not executed.'}
(root/'LOCAL_VALIDATION_09.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
