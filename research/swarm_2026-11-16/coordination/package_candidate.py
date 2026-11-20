from pathlib import Path
import argparse, ast, hashlib, json, os, subprocess, tempfile

BASE = '297e54fd105e974cdf401e5b4d20253a22c16f90'
p = argparse.ArgumentParser()
p.add_argument('--sources', required=True)
p.add_argument('--out', required=True)
p.add_argument('--repo', default='/mnt/d/OpenKernelChallenge/serving/sglang')
a = p.parse_args()
sources, out, repo = Path(a.sources), Path(a.out), Path(a.repo)
out.mkdir(parents=True, exist_ok=True)
def git(args, cwd, check=True):
    r = subprocess.run(['git', *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and r.returncode:
        raise RuntimeError((r.stderr + r.stdout).decode(errors='replace'))
    return r
files = {}
for source in sorted(sources.rglob('*')):
    if not source.is_file() or '__pycache__' in source.parts or source.suffix == '.pyc':
        continue
    rel = source.relative_to(sources).as_posix()
    assert not source.is_symlink(), rel
    assert rel.startswith('python/sglang/'), rel
    assert '/srt/model_loader/' not in rel, rel
    assert source.suffix in ('.py', '.json', '.cu', '.cuh', '.h', '.cpp', '.cc'), rel
    data = source.read_bytes()
    if source.suffix == '.py':
        ast.parse(data.decode(), filename=rel)
    elif source.suffix == '.json':
        json.loads(data)
    files[rel] = data
assert files
with tempfile.TemporaryDirectory(prefix='okc_package_') as td:
    temp = Path(td)
    git(['init', '-q'], temp)
    old_files = {}
    for rel in files:
        old = git(['show', f'{BASE}:{rel}'], repo, check=False)
        if old.returncode == 0:
            dest = temp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(old.stdout)
            old_files[rel] = old.stdout
        else:
            exists = git(['ls-tree', BASE, '--', rel], repo).stdout
            assert not exists, (rel, old.stderr.decode())
    git(['add', '-A'], temp)
    baseline_tree = git(['write-tree'], temp).stdout.decode().strip()
    for rel, data in files.items():
        dest = temp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    git(['add', '-A'], temp)
    expected_tree = git(['write-tree'], temp).stdout.decode().strip()
    patch = git(['diff', '--cached', '--no-renames', '--full-index', baseline_tree], temp).stdout
    assert len(patch) <= 400000
    added = [line[1:] for line in patch.decode().splitlines() if line.startswith('+') and not line.startswith('+++')]
    assert max(map(len, added), default=0) <= 400
    dest = out / 'candidate.diff'
    dest.write_bytes(patch)
    git(['diff', '--cached', '--check', baseline_tree], temp)
    git(['apply', '--reverse', '--cached', '--check', str(dest)], temp)
    git(['apply', '--reverse', '--cached', str(dest)], temp)
    assert git(['write-tree'], temp).stdout.decode().strip() == baseline_tree
    git(['apply', '--cached', '--check', str(dest)], temp)
    git(['apply', '--cached', str(dest)], temp)
    assert git(['write-tree'], temp).stdout.decode().strip() == expected_tree
    summary = git(['apply', '--summary', str(dest)], temp).stdout.decode()
    assert all(line.startswith(' create mode 100644 ') for line in summary.splitlines()), summary
    manifest = {
        'base': BASE, 'patch_sha256': hashlib.sha256(patch).hexdigest(),
        'patch_bytes': len(patch), 'file_count': len(files),
        'max_added_line_chars': max(map(len, added), default=0),
        'source_hashes': {rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()},
        'new_files': [rel for rel in files if rel not in old_files],
        'git_summary': summary, 'roundtrip_apply_exact': True,
        'python_ast': 'passed', 'whitespace_check': 'passed',
        'original_checkout_modified': False,
        'validation_scope': 'Package/source validation only; GPU and serving evidence separate.'
    }
    (out / 'package_validation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
