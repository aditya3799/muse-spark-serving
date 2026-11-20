import hashlib,json,sys,subprocess
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from ownership_study import OwnershipStudy
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest_07.json').read_text());helper=(root/'mapping_helper.py').read_text()
cpu=OwnershipStudy(manifest,helper,output_root=root,force_cpu=True,samples=5,repeats=1,trials=1)
cpu.environment();assert cpu.lifecycle()['status']=='passed'
assert cpu.benchmark()['status']=='skipped';assert cpu.profile()['status']=='skipped';cpu.export()
checkout=root/'patch_validation';target=checkout/'python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py';target.parent.mkdir(parents=True,exist_ok=True)
baseline=Path(r'D:\OpenKernelChallenge\serving\sglang\python\sglang\srt\layers\muse_spark\muse_spark_fp4_moe.py').read_text(encoding='utf-8')
target.write_text(baseline,encoding='utf-8',newline='\n')
subprocess.run(['git','apply','--check',str(root/'single_owner_v4_draft.diff')],cwd=checkout,capture_output=True,check=True)
subprocess.run(['git','apply',str(root/'single_owner_v4_draft.diff')],cwd=checkout,capture_output=True,check=True)
assert target.read_text()==(root/'single_owner_muse_spark_fp4_moe.py').read_text()
path=root/'07_kaggle_single_owner.ipynb'
if not path.exists():path=root.parent/path.name
nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:client.execute()
finally:
    nbformat.write(nb,root/'local_validation_07.executed.ipynb')
    if manager.has_kernel:manager.shutdown_kernel(now=True)
reports=sorted(root.glob('v4_single_owner_*/report.json'),key=lambda p:p.stat().st_mtime)
r=json.loads(reports[-1].read_text())
for section in ('environment','lifecycle','benchmark'):assert r['sections'][section]['status']=='passed',r['sections'][section]
assert r['sections']['benchmark']['changed_input_schedule_checks']==360
assert r['totals']=={'cases':30,'timing_rows':180,'raw_samples':1980}
for name,h in json.loads((reports[-1].parent/'checksums.json').read_text()).items():assert hashlib.sha256((reports[-1].parent/name).read_bytes()).hexdigest()==h
result={'schema':'passed','cpu_lifecycle_and_export':'passed','draft_applies_in_temporary_tree':True,'active_v4_modified':False,
    'statuses':{k:v['status'] for k,v in r['sections'].items()},'device':r['sections']['environment']['gpu_devices'],
    'lifecycle':r['sections']['lifecycle'],'benchmark':r['sections']['benchmark'],'totals':r['totals'],
    'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_path':str(reports[-1]),
    'note':'Local source-slice validation; registered-buffer introspection compatibility and actual fused-kernel behavior with map offset 1 remain untested.'}
(root/'LOCAL_VALIDATION_07.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
