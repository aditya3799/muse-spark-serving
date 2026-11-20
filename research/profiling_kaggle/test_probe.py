import hashlib
import json
import zipfile
from pathlib import Path
from probe_support import Preflight, make_routes, route_oracle


def test_oracle_known_padding_and_sentinels():
    got = route_oracle([64, 65, -1, 200, 64], block=2)
    assert got['local'] == [0, 1, -1, -1, 0]
    assert got['packed'] == [2, 3, 0, 4, 1, 5]
    assert got['experts'] == [-1, 0, 1]
    assert got['counts'][:3] == [2, 2, 1]
    assert got['padded'][:3] == [2, 2, 2]


def test_route_generation_and_accounting():
    for m in (1, 8, 9, 32, 128, 256):
        for mode in ('uniform', 'hot_local', 'all_remote', 'all_local', 'sentinel'):
            values = make_routes(m, mode)
            assert len(values) == m * 8
            for i in range(m):
                row = values[i * 8:(i + 1) * 8]
                assert len(set(row)) == 8
            r = route_oracle(values)
            assert sum(r['counts']) == len(values)
            assert len(r['packed']) == sum(r['padded'])
            assert sorted(x for x in r['packed'] if x < len(values)) == list(range(len(values)))
            if mode == 'all_remote':
                assert r['counts'][0] == len(values)
            if mode == 'all_local':
                assert r['counts'][0] == 0


def test_cpu_failure_tolerant_export(tmp_path):
    p = Preflight({}, '', output_root=tmp_path, force_cpu=True)
    p.environment()
    assert p.timing_and_graphs()['status'] == 'skipped'
    assert p.dtype_checks()['status'] == 'skipped'
    assert p.original_kernel_probe()['status'] == 'skipped'
    assert p.routing_diagnostic()['structural_cases'] == 30
    assert p.profile()['status'] == 'skipped'
    p.section('expected_failure', lambda: 1 / 0)
    archive = p.export()
    with zipfile.ZipFile(archive) as z:
        report = json.loads(z.read('report.json'))
        assert report['sections']['expected_failure']['status'] == 'failed'
        assert 'Missing sections:' not in z.read('SUMMARY.txt').decode()
        for name, checksum in json.loads(z.read('checksums.json')).items():
            assert hashlib.sha256(z.read(name)).hexdigest() == checksum


def test_t4_skips_original_kernel_before_import_or_execution(tmp_path):
    from types import SimpleNamespace
    p = Preflight({}, 'must never be executed', output_root=tmp_path)
    p.gpu = True
    p.torch = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: (7, 5)))
    result = p.original_kernel_probe()
    assert result['status'] == 'skipped'
    assert 'SM80' in result['reason']
    assert not list(p.out.glob('v4_alignment_probe.py'))
