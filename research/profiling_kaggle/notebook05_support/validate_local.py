import hashlib,json,sys,subprocess
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from source_study import SourceStudy

root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest_05.json').read_text());helper=(root/'mapping_helper.py').read_text()
cpu=SourceStudy(manifest,helper,output_root=root,force_cpu=True,samples=5,repeats=1,trials=1)
cpu.environment();assert cpu.source_checks()['status']=='passed'
assert cpu.benchmark()['status']=='skipped';assert cpu.cadence()['status']=='skipped';assert cpu.profile()['status']=='skipped';cpu.export()
# Verify the diff applies cleanly against the exact V4 file in a temporary tree.
checkout=root/'patch_validation';target=checkout/'python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
target.parent.mkdir(parents=True,exist_ok=True)
source=Path(r'D:\OpenKernelChallenge\serving\sglang\python\sglang\srt\layers\muse_spark\muse_spark_fp4_moe.py').read_text(encoding='utf-8')
target.write_text(source,encoding='utf-8',newline='\n')
subprocess.run(['git','apply','--check',str(root/'cached_mapping_v4_draft.diff')],cwd=checkout,check=True,capture_output=True)
subprocess.run(['git','apply',str(root/'cached_mapping_v4_draft.diff')],cwd=checkout,check=True,capture_output=True)
assert target.read_text()==(root/'candidate_muse_spark_fp4_moe.py').read_text()
path=root/'05_kaggle_source_candidate.ipynb'
if not path.exists():path=root.parent/path.name
nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:client.execute()
finally:
    nbformat.write(nb,root/'local_validation_05.executed.ipynb')
    if manager.has_kernel:manager.shutdown_kernel(now=True)
reports=sorted(root.glob('v4_source_candidate_*/report.json'),key=lambda p:p.stat().st_mtime)
r=json.loads(reports[-1].read_text())
for section in ('environment','source_checks','benchmark','update_cadence'):assert r['sections'][section]['status']=='passed',r['sections'][section]
assert r['totals']=={'cases':66,'timing_rows':264,'raw_samples':2904}
assert r['sections']['source_checks']['input_cases']==672
assert r['sections']['source_checks']['deferred_fused_cases']==96
for name,h in json.loads((reports[-1].parent/'checksums.json').read_text()).items():assert hashlib.sha256((reports[-1].parent/name).read_bytes()).hexdigest()==h
result={'schema':'passed','cpu_source_checks':'passed','draft_applies_to_exact_v4_in_temporary_tree':True,
        'live_source_edited':False,'statuses':{k:v['status'] for k,v in r['sections'].items()},'source_checks':r['sections']['source_checks'],
        'device':r['sections']['environment']['gpu_devices'],'totals':r['totals'],
        'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_path':str(reports[-1]),
        'note':'Source-slice correctness and local timings only. Actual fused alignment, expert GEMMs, full model and H200 remain untested.'}
(root/'LOCAL_VALIDATION_05.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
