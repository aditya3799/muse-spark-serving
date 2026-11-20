import hashlib
import json
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from integration_study import IntegrationStudy

root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest_04.json').read_text())
helper=(root/'mapping_helper.py').read_text()
cpu=IntegrationStudy(manifest,helper,output_root=root,force_cpu=True,samples=5,repeats=1,trials=1)
cpu.environment()
assert cpu.lifecycle()['status']=='passed'
assert cpu.benchmark()['status']=='skipped'
assert cpu.update_cost()['status']=='skipped'
assert cpu.profile()['status']=='skipped'
cpu.export()
for name,digest in json.loads((cpu.out/'checksums.json').read_text()).items():assert hashlib.sha256((cpu.out/name).read_bytes()).hexdigest()==digest
path=root/'04_kaggle_mapping_integration.ipynb'
if not path.exists():path=root.parent/path.name
nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:client.execute()
finally:
    nbformat.write(nb,root/'local_validation_04.executed.ipynb')
    if manager.has_kernel:manager.shutdown_kernel(now=True)
reports=sorted(root.glob('v4_mapping_integration_*/report.json'),key=lambda p:p.stat().st_mtime)
report=json.loads(reports[-1].read_text())
for key in ('environment','lifecycle','benchmark','update_cost'):assert report['sections'][key]['status']=='passed',report['sections'][key]
assert report['sections']['benchmark']['cases']==48
assert report['sections']['benchmark']['changed_input_graph_checks']==288
assert report['totals']=={'timing_rows':204,'raw_samples':2244}
for name,digest in json.loads((reports[-1].parent/'checksums.json').read_text()).items():assert hashlib.sha256((reports[-1].parent/name).read_bytes()).hexdigest()==digest
result={'cpu_lifecycle_and_export':'passed','schema':'passed','statuses':{k:v['status'] for k,v in report['sections'].items()},
    'device':report['sections']['environment']['gpu_devices'],'lifecycle':report['sections']['lifecycle'],'benchmark':report['sections']['benchmark'],
    'totals':report['totals'],'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'report_path':str(reports[-1]),'note':'Local validation of a prototype, not full model or Kaggle T4 performance.'}
(root/'LOCAL_VALIDATION_04.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
