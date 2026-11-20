"""Isolated candidate assembly. Never changes the user's checkout or Git index."""
import ast, hashlib, json, os, pathlib, subprocess
R=pathlib.Path('/mnt/d/OpenKernelChallenge/serving'); O=R/'research_2026-11-19_final'
PIN='297e54fd105e974cdf401e5b4d20253a22c16f90'
BASE='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'
assert hashlib.sha256((R/'release_2026-11-18/candidate.diff').read_bytes()).hexdigest()==BASE
E=dict(os.environ,GIT_INDEX_FILE=str(O/'assembly.index'))
def git(*args):return subprocess.check_output(['git','-C',str(R/'sglang'),*args],env=E)
def put(path,text):
    ast.parse(text) if path.endswith('.py') else None
    dest=O/'assembled'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
    sha=git('hash-object','-w',str(dest)).decode().strip()
    git('update-index','--add','--cacheinfo','100644',sha,path)
def get(path):return git('show',':'+path).decode()
def replace(text,old,new):
    assert text.count(old)==1,(old[:100],text.count(old));return text.replace(old,new)
git('read-tree',PIN)
patches=['research_2026-11-19/prefill/integration_joint/candidate.diff',
 'research_2026-11-19_round2/coordination/projection_failure_policy/incremental.diff',
 'research_2026-11-19_round2/prefill/early_projection_unapplied/incremental.diff',
 'research_2026-11-19_round2/prefill/row_no_grad_unapplied/incremental.diff',
 'research_2026-11-19_round2/prefill/row_constructor_cleanup_unapplied/incremental.diff']
for p in patches:git('apply','--cached',str(R/p))
prefix='python/sglang/srt/speculative/'
gate='''"""Conservative paired startup admission, not a serving-speed prediction."""
import math
import statistics

def latency_gate(samples, minimum_gain=1.10):
    native=samples.get('native_us', [])
    candidate=samples.get('combined_us', [])
    control=samples.get('control_us', [])
    if len(native)<12 or len(native)!=len(candidate) or len(native)!=len(control):
        return {'accepted':False, 'reason':'insufficient paired controls'}
    if any(not math.isfinite(x) or x<=0 for row in (native,candidate,control) for x in row):
        return {'accepted':False, 'reason':'nonpositive or nonfinite timing'}
    ratios=[min(a,b)/c for a,b,c in zip(native,control,candidate)]
    controls=[a/b for a,b in zip(native,control)]
    halves=[statistics.median(ratios[:len(ratios)//2]),statistics.median(ratios[len(ratios)//2:])]
    # Every pair must improve; each half independently clears the margin.
    # Duplicate native controls must agree. Deliberately fail closed on noise.
    stable=all(0.95<=r<=1.05 for r in controls)
    accepted=stable and min(ratios)>1.0 and min(halves)>minimum_gain
    return {'accepted':accepted,'minimum_pair_gain':min(ratios),
            'median_gain':statistics.median(ratios),'half_median_gains':halves,
            'control_range':[min(controls),max(controls)],'controls_stable':stable,
            'required_gain':minimum_gain,'reason':'paired startup admission only'}
'''
put(prefix+'dflash_projection_gate.py',gate)
row=get(prefix+'dflash_projection_row.py')
row=replace(row,'import statistics\n','import statistics\nfrom sglang.srt.speculative.dflash_projection_gate import latency_gate\n')
row=replace(row,'def _row_startup_timing(router,m,k,generator,rounds=4):','def _row_startup_timing(router,m,k,generator,rounds=12):')
row=replace(row,"('native_us','local_us','gather_us','combined_us')","('native_us','control_us','local_us','gather_us','combined_us')")
row=replace(row,"    for step in range(-1,rounds):\n        for arm in (('native','candidate') if step%2==0 else ('candidate','native')):",
"    orders=(('native','candidate','control'),('control','candidate','native'),\n            ('candidate','native','control'),('control','native','candidate'),\n            ('native','control','candidate'),('candidate','control','native'))\n    for step in range(-2,rounds):\n        for arm in orders[step%len(orders)]:")
row=replace(row,"            if arm=='native':\n                try:","            if arm in ('native','control'):\n                try:")
row=replace(row,"                if arm=='native':records['native_us'].append(events[0].elapsed_time(events[2])*1000)",
"                if arm in ('native','control'):records[arm+'_us'].append(events[0].elapsed_time(events[2])*1000)")
row=replace(row,"                faster=all(t.get('active') and t['medians']['combined_us']<t['medians']['native_us'] for t in timings)",
"                for t in timings:\n                    t['admission']=latency_gate(t.get('samples',{}))\n                faster=all(t.get('active') and t['admission']['accepted'] for t in timings)")
put(prefix+'dflash_projection_row.py',row)
entry=get(prefix+'dflash_projection_n/entry.py')
entry=replace(entry,'import json,statistics','import json,statistics\nfrom sglang.srt.speculative.dflash_projection_gate import latency_gate')
entry=replace(entry,'def _startup_timing(router,x,rounds=6):','def _startup_timing(router,x,rounds=12):')
entry=replace(entry,"records={'native_us':[],'local_us':[],'gather_us':[],'combined_us':[]};bad=0;fallbacks=0",
"records={'native_us':[],'control_us':[],'local_us':[],'gather_us':[],'combined_us':[]};bad=0;fallbacks=0")
entry=replace(entry," for step in range(rounds):\n  for arm in (('native','candidate') if step%2==0 else ('candidate','native')):",
" orders=(('native','candidate','control'),('control','candidate','native'),\n         ('candidate','native','control'),('control','native','candidate'),\n         ('native','control','candidate'),('candidate','control','native'))\n for step in range(rounds):\n  for arm in orders[step%len(orders)]:")
entry=replace(entry,"   if arm=='native':out=router.original(x);events[2].record()","   if arm in ('native','control'):out=router.original(x);events[2].record()")
entry=replace(entry,"   if arm=='native':records['native_us'].append(events[0].elapsed_time(events[2])*1000)",
"   if arm in ('native','control'):records[arm+'_us'].append(events[0].elapsed_time(events[2])*1000)")
entry=replace(entry,"   bad+=int(not torch.equal(out.view(torch.int16),reference.view(torch.int16)))",
"   bad+=int(not torch.equal(out.view(torch.int16),reference.view(torch.int16)))\n   del out")
entry=replace(entry,"  faster=not measure or (metadata.get('globally_enabled') and all(r['timing'] is not None and r['timing'].get('active') and r['timing'].get('correct') and r['timing']['medians']['combined_us']<r['timing']['medians']['native_us'] for r in reports))",
"  for r in reports:\n   if r['timing'] is not None:r['timing']['admission']=latency_gate(r['timing'].get('samples',{}))\n  faster=not measure or (metadata.get('globally_enabled') and all(r['timing'] is not None and r['timing'].get('active') and r['timing'].get('correct') and r['timing']['admission']['accepted'] for r in reports))")
put(prefix+'dflash_projection_n/entry.py',entry)
git('diff','--cached','--check')
paths=git('diff','--cached','--name-only',PIN).decode().splitlines()
for p in paths:
    data=git('show',':'+p);dest=O/'assembled'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    if p.endswith('.py'):ast.parse(data)
worker=prefix+'dflash_worker_v2.py'
original_worker=get(worker)
original_row=get(prefix+'dflash_projection_row.py')
variants={}
for name,row_on,n_on in [('row_only',True,False),('decode_only',False,True),('joint',True,True),('disabled_control',False,False)]:
    put(worker,replace(original_worker,'os.environ.get("SGLANG_EXPERIMENTAL_DFLASH_OUTPUT_N", "1")','os.environ.get("SGLANG_EXPERIMENTAL_DFLASH_OUTPUT_N", "'+str(int(n_on))+'")'))
    put(prefix+'dflash_projection_row.py',replace(original_row,"os.environ.get('SGLANG_EXPERIMENTAL_DFLASH_ROW_FC','1')","os.environ.get('SGLANG_EXPERIMENTAL_DFLASH_ROW_FC','"+str(int(row_on))+"')"))
    git('diff','--cached','--check')
    patch=git('diff','--cached','--binary',PIN)
    assert len(patch)<400*1024
    added=[line[1:] for line in patch.decode().splitlines() if line.startswith('+') and not line.startswith('+++')]
    assert max(map(len,added))<=400
    d=O/'candidates'/name;d.mkdir(parents=True,exist_ok=True);(d/'candidate.diff').write_bytes(patch)
    (d/'flags').write_text('--moe-runner-backend humming\n')
    variants[name]={'sha256':hashlib.sha256(patch).hexdigest(),'bytes':len(patch),'row':row_on,'decode':n_on}
put(worker,original_worker);put(prefix+'dflash_projection_row.py',original_row)
(O/'CANDIDATES.json').write_text(json.dumps({'pin':PIN,'parent_sha256':BASE,'patches':patches,'variants':variants,'scope':'Unsubmitted research candidates; not H200 validated'},indent=2))
print(json.dumps(variants,indent=2))
