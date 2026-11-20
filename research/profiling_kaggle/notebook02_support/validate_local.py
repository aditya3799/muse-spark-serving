import ast
import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from routing_study import RoutingStudy, cpu_reference
import routing_study

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'source_manifest_02.json').read_text())
helper = (root / 'mapping_helper.py').read_text()
assert hashlib.sha256((root / 'routing_study.py').read_bytes()).hexdigest() == manifest['support_sha256']
# Independent known example: a local ID, inactive sentinel, remote ID and another local ID.
mapping = [-1] * 256
mapping[64:96] = list(range(32))
known = cpu_reference([64, -1, 0, 95], mapping)
assert known['original_mapping_i32'] == ([0, -1, -1, 31],)
assert known['sort_and_count'][0] == [1, 2, 0, 3]
assert known['sort_and_count'][1][0] == 2
assert sum(known['padding_counts'][0]) == 24
cpu = RoutingStudy(manifest, helper, output_root=root, force_cpu=True, samples=5, repeats=1)
cpu.environment()
assert cpu.correctness()['status'] == 'passed'
assert cpu.benchmark()['status'] == 'skipped'
assert cpu.profile()['status'] == 'skipped'
try:
    cpu.check(cpu.torch.tensor([0, 1]), ([0, 2],))
except AssertionError:
    pass
else:
    raise AssertionError('Correctness checker accepted corrupted output')
archive = cpu.export()
assert archive.is_file()
for name, digest in json.loads((cpu.out / 'checksums.json').read_text()).items():
    assert hashlib.sha256((cpu.out / name).read_bytes()).hexdigest() == digest

notebook_path = root / '02_kaggle_routing_overhead.ipynb'
if not notebook_path.exists():
    notebook_path = root.parent / notebook_path.name
nb = nbformat.read(notebook_path, as_version=4)
nbformat.validate(nb)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(nb, km=manager, timeout=600, resources={'metadata': {'path': str(root)}}, allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(nb, root / 'local_validation_02.executed.ipynb')
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
reports = sorted(root.glob('v4_routing_overhead_*/report.json'), key=lambda p: p.stat().st_mtime)
report = json.loads(reports[-1].read_text())
assert report['sections']['correctness']['status'] == 'passed'
assert report['sections']['benchmark']['status'] == 'passed', report['sections']['benchmark']
assert report['sections']['benchmark']['cases'] == 50
assert report['sections']['benchmark']['timing_rows'] == 110
assert report['sections']['benchmark']['raw_samples'] == 1650
for name, digest in json.loads((reports[-1].parent / 'checksums.json').read_text()).items():
    assert hashlib.sha256((reports[-1].parent / name).read_bytes()).hexdigest() == digest
validation = {'notebook_schema': 'passed', 'cpu_oracle_known_example': 'passed',
              'cpu_fallback_export_checksums': 'passed', 'corrupted_output_rejected': True,
              'executed_code_cells': sum(c.cell_type == 'code' and c.execution_count is not None for c in nb.cells),
              'local_device': report['sections']['environment']['gpu_devices'],
              'statuses': {k: v['status'] for k, v in report['sections'].items()},
              'benchmark': report['sections']['benchmark'],
              'notebook_sha256': hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              'note': 'Local validation, not Kaggle timing evidence. Kaggle T4 validation remains pending.',
              'executed_notebook': str(root / 'local_validation_02.executed.ipynb'), 'report_path': str(reports[-1])}
# The local Windows profiler lacks device events. Exercise the exact trace-parser
# statements on our downloaded T4 trace without invoking or mocking CUDA execution.
kaggle_trace = Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle\routing_trace.json')
if kaggle_trace.exists():
    tree = ast.parse((root / 'routing_study.py').read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RoutingStudy')
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'profile')
    run = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    first = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'events' for t in n.targets))
    wrapper = ast.parse('def parse_existing(self, path):\n    pass\n').body[0]
    wrapper.body = run.body[first:]
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = dict(vars(routing_study))
    exec(compile(module, '<existing-trace-parser-check>', 'exec'), namespace)
    out = root / 'existing_trace_validation'
    out.mkdir(exist_ok=True)
    result = namespace['parse_existing'](SimpleNamespace(out=out), kaggle_trace)
    assert result['status'] == 'passed', result
    assert result['gpu_kernel_count'] == 102
    assert result['device_copy_count'] == 18
    assert result['unattributed_device_operations'] == 0
    validation['existing_kaggle_trace_parser'] = result
(root / 'LOCAL_VALIDATION_02.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
print(json.dumps(validation, indent=2))
