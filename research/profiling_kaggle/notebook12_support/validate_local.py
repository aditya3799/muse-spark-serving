import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from gate_study import GateStudy
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest_12.json').read_text());helper=(root/'mapping_helper.py').read_text()
cpu=GateStudy(manifest,helper,output_root=root,force_cpu=True)
cpu.environment();assert cpu.matrix()['status']=='passed';assert cpu.contracts_gate()['status']=='passed'
assert cpu.update_gate()['status']=='skipped';assert cpu.moves_gate()['status']=='skipped';cpu.export()
assert cpu.report['sections']['matrix']['outputs_checked']==2592
assert cpu.report['sections']['matrix']['deferred_fused_metadata_outputs']==0
checkout=root/'patch_validation';target=checkout/'python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py';target.parent.mkdir(parents=True,exist_ok=True)
baseline=Path(r'D:\OpenKernelChallenge\serving\sglang\python\sglang\srt\layers\muse_spark\muse_spark_fp4_moe.py').read_text(encoding='utf-8')
assert hashlib.sha256(baseline.encode()).hexdigest()==manifest['baseline_owner_sha256']
target.write_text(baseline,encoding='utf-8',newline='\n')
subprocess.run(['git','apply','--check',str(root/'selected_two_buffer_v4.diff')],cwd=checkout,capture_output=True,check=True)
subprocess.run(['git','apply',str(root/'selected_two_buffer_v4.diff')],cwd=checkout,capture_output=True,check=True)
assert target.read_text(encoding='utf-8')==(root/'selected_two_buffer_source.py').read_text(encoding='utf-8')
path=root/'12_kaggle_two_buffer_gate.ipynb';nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:client.execute()
finally:
    nbformat.write(nb,root/'local_validation_12.executed.ipynb')
    if manager.has_kernel:manager.shutdown_kernel(now=True)
reports=sorted(root.glob('v4_two_buffer_gate_*/report.json'),key=lambda p:p.stat().st_mtime);r=json.loads(reports[-1].read_text())
assert r['gate_status']=='passed'
for section in ('environment','matrix','updates','moves','contracts'):assert r['sections'][section]['status']=='passed',r['sections'][section]
m=r['sections']['matrix'];assert (m['input_cases'],m['outputs_checked'],m['deferred_fused_metadata_outputs'],m['value_checked_outputs'])==(1296,2592,384,2208)
assert r['sections']['updates']['graph_output_checks']==128 and r['sections']['updates']['stable_pointer_checks']==64
assert r['sections']['moves']['output_checks']==48 and r['sections']['contracts']['invalid_updates_rejected_before_mutation']==25
with zipfile.ZipFile(reports[-1].parent.with_suffix('.zip')) as z:
    hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
result={'schema':'passed','cpu_matrix_and_contracts':'passed','selected_patch_applies_in_temporary_tree':True,'active_v4_modified':False,
    'statuses':{k:v['status'] for k,v in r['sections'].items()},'device':r['sections']['environment']['gpu_devices'],
    'matrix':m,'updates':r['sections']['updates'],'moves':r['sections']['moves'],'contracts':r['sections']['contracts'],
    'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_path':str(reports[-1]),
    'note':'Correctness gate only; metadata-only fused branch checks do not validate the actual omitted kernel. Full integration remains untested.'}
(root/'LOCAL_VALIDATION_12.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
