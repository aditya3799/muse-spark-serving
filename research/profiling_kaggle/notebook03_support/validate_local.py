import hashlib
import json
import sys
import zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from mapping_study import MappingStudy, reference, parse_trace

root = Path(__file__).resolve().parent
manifest = json.loads((root/'source_manifest_03.json').read_text())
helper = (root/'mapping_helper.py').read_text()
assert hashlib.sha256((root/'mapping_study.py').read_bytes()).hexdigest()==manifest['support_sha256']
assert reference([-2147483648,-2,-1,0,2], [7,-1,3])==[-1,-1,-1,7,3]
cpu = MappingStudy(manifest,helper,output_root=root,force_cpu=True,samples=5,repeats=1,trials=1)
cpu.environment()
assert cpu.correctness()['status']=='passed'
assert cpu.benchmark()['status']=='skipped'
assert cpu.setup_cost()['status']=='skipped'
assert cpu.profile()['status']=='skipped'
try:
    cpu.check(cpu.t.tensor([[0]*8],dtype=cpu.t.int32),[1]*8,1)
except AssertionError:
    pass
else:
    raise AssertionError('Corrupted output accepted')
cpu.export()
for name,h in json.loads((cpu.out/'checksums.json').read_text()).items():
    assert hashlib.sha256((cpu.out/name).read_bytes()).hexdigest()==h

# Use real T4 launch timestamps/correlations, changing only annotation prefixes
# from notebook-02 diagnostic/ to this notebook's mapping/ naming convention.
existing = Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle\v4_routing_overhead_20261114T143842Z_926fce.zip')
parser_validation = {'status':'not_run'}
if existing.exists():
    with zipfile.ZipFile(existing) as z: trace = json.loads(z.read('routing_trace.json'))
    for event in trace['traceEvents']:
        if event.get('cat')=='user_annotation':
            event['name']=event['name'].replace('diagnostic/','mapping/',1)
    out = root/'parser_validation'
    out.mkdir(exist_ok=True)
    path = out/'adapted_existing_trace.json'
    path.write_text(json.dumps(trace))
    parser_validation = parse_trace(path,out)
    assert parser_validation['status']=='passed'
    assert parser_validation['trace_categories']['kernel']==102
    assert parser_validation['unattributed_device_operations']==0

path = root/'03_kaggle_mapping_candidate.ipynb'
if not path.exists(): path = root.parent/path.name
nb = nbformat.read(path,as_version=4)
nbformat.validate(nb)
assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv[0]=sys.executable
client = NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(root)}},allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(nb,root/'local_validation_03.executed.ipynb')
    if manager.has_kernel: manager.shutdown_kernel(now=True)
reports = sorted(root.glob('v4_mapping_candidate_*/report.json'),key=lambda p:p.stat().st_mtime)
report = json.loads(reports[-1].read_text())
for section in ('correctness','benchmark','setup_cost'):
    assert report['sections'][section]['status']=='passed', report['sections'][section]
assert report['sections']['benchmark']['cases']==96
assert report['sections']['correctness']['input_cases']==672
assert report['sections']['correctness']['output_checks']==4032
assert report['sections']['benchmark']['changed_input_graph_checks']==1152
import csv
with (reports[-1].parent/'timings.csv').open() as f: rows=list(csv.DictReader(f))
with (reports[-1].parent/'raw_samples.csv').open() as f: raw=list(csv.DictReader(f))
assert len(rows)==588 and len(raw)==6468
for name,h in json.loads((reports[-1].parent/'checksums.json').read_text()).items():
    assert hashlib.sha256((reports[-1].parent/name).read_bytes()).hexdigest()==h
result = {'notebook_schema':'passed','cpu_reference_known_case':'passed','cpu_fallback_and_export':'passed',
          'corrupted_output_rejected':True,'existing_t4_trace_parser':parser_validation,
          'executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),
          'statuses':{k:v['status'] for k,v in report['sections'].items()},
          'device':report['sections']['environment']['gpu_devices'],
          'correctness':report['sections']['correctness'],'benchmark':report['sections']['benchmark'],
          'total_timing_rows':len(rows),'total_raw_samples':len(raw),
          'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'report_path':str(reports[-1]),'note':'Local validation only. Kaggle T4 candidate performance remains to be measured.'}
(root/'LOCAL_VALIDATION_03.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
