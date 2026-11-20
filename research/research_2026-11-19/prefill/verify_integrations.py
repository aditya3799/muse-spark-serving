"""Validate cumulative research patches against an isolated frozen-pin index."""
import ast,hashlib,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PIN='297e54fd105e974cdf401e5b4d20253a22c16f90'
results=[]
for name in ('integration_row','integration_joint'):
    folder=HERE/name;manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['pin']==PIN
    assert hashlib.sha256((folder/'candidate.diff').read_bytes()).hexdigest()==manifest['candidate_sha256']
    for record in manifest['files']:
        path=folder/'sources'/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        if path.suffix=='.py':ast.parse(path.read_text())
    env=dict(os.environ,GIT_INDEX_FILE=str(folder/'validation.index'))
    subprocess.run(['git','-C',str(ROOT/'sglang'),'read-tree',PIN],env=env,check=True)
    subprocess.run(['git','-C',str(ROOT/'sglang'),'apply','--cached','--check',str(folder/'candidate.diff')],env=env,check=True)
    results.append(dict(name=name,candidate_sha256=manifest['candidate_sha256'],syntax_and_manifest=True,cumulative_apply_against_frozen_pin=True,
                        row_enabled=manifest['row_enabled'],n_enabled=manifest['n_enabled'],working_tree_modified=False))
report=dict(complete=True,results=results)
(HERE/'integration_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
