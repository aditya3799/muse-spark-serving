import hashlib,json,sys,zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from stream_study import StreamStudy
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest_13.json').read_text());helper=(root/'mapping_helper.py').read_text()
cpu=StreamStudy(manifest,helper,output_root=root,force_cpu=True)
cpu.environment();assert cpu.smoke()['status']=='passed';assert cpu.contracts_gate()['status']=='passed'
assert cpu.handoff()['status']=='skipped';cpu.export()
path=root/'13_kaggle_stream_ordering.ipynb';nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:client.execute()
finally:
    nbformat.write(nb,root/'local_validation_13.executed.ipynb')
    if manager.has_kernel:manager.shutdown_kernel(now=True)
reports=sorted(root.glob('v4_stream_ordered_*/report.json'),key=lambda p:p.stat().st_mtime);r=json.loads(reports[-1].read_text())
assert r['gate_status']=='passed'
for section in ('environment','smoke','contracts','handoff'):assert r['sections'][section]['status']=='passed',r['sections'][section]
h=r['sections']['handoff'];assert (h['handoff_cycles'],h['output_checks'],h['stable_pointer_checks'],h['partitions_recaptured_after_move'])==(768,1536,768,8)
assert not h['device_wide_sync_in_handoff'] and h['intentionally_unsynchronized_cases']==0
with zipfile.ZipFile(reports[-1].parent.with_suffix('.zip')) as z:
    hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,digest in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==digest
result={'schema':'passed','cpu_prerequisites':'passed','active_v4_modified':False,
    'statuses':{k:v['status'] for k,v in r['sections'].items()},'device':r['sections']['environment']['gpu_devices'],'handoff':h,
    'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_path':str(reports[-1]),
    'note':'Correct event-ordered source-slice schedule only. No production integration, throughput, actual fused kernel or full-model validation.'}
(root/'LOCAL_VALIDATION_13.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
