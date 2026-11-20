import ast, copy, hashlib, importlib.util, json, math, os, pathlib, subprocess, types
R=pathlib.Path('/mnt/d/OpenKernelChallenge/serving');O=R/'research_2026-11-19_final'
prefix='python/sglang/srt/speculative/'
scope={};exec((O/'assembled'/prefix/'dflash_projection_gate.py').read_text(),scope);gate=scope['latency_gate']
good={'native_us':[100.]*12,'control_us':[100.]*12,'combined_us':[70.]*12}
cases=[('clear_gain',good,True)]
for name,field,value in [('regression','combined_us',110.),('tie','combined_us',100.),('small_gain','combined_us',95.),('noisy_control','control_us',90.),('zero','combined_us',0.),('nan','native_us',float('nan')),('inf','control_us',float('inf'))]:
    data=copy.deepcopy(good);data[field][5]=value
    if name=='small_gain':data[field]=[value]*12
    cases.append((name,data,False))
data=copy.deepcopy(good);data['combined_us'][6:]=[97.]*6;cases.append(('holdout_half_loses_margin',data,False))
data=copy.deepcopy(good);data['control_us']=data['control_us'][:11];cases.append(('unpaired',data,False))
cases.append(('missing',{},False))
tests=[]
for name,data,expect in cases:
    result=gate(data);assert result['accepted']==expect,(name,result)
    tests.append({'name':name,'passed':True,'result':result})
env=dict(os.environ,GIT_INDEX_FILE=str(O/'validation.index'))
def git(*args):return subprocess.check_output(['git','-C',str(R/'sglang'),*args],env=env)
PIN='297e54fd105e974cdf401e5b4d20253a22c16f90'
validation={}
for directory in (O/'candidates').iterdir():
    patch=directory/'candidate.diff';git('read-tree',PIN);git('apply','--cached',str(patch));git('diff','--cached','--check')
    paths=git('diff','--cached','--name-only',PIN).decode().splitlines()
    parsed=0
    for path in paths:
        assert path.startswith('python/sglang/') and not path.startswith('python/sglang/srt/model_loader/')
        if path.endswith('.py'):ast.parse(git('show',':'+path));parsed+=1
    validation[directory.name]={'apply':True,'whitespace':True,'allowed_paths':True,'python_files_parsed':parsed,'sha256':hashlib.sha256(patch.read_bytes()).hexdigest()}
# Verify the new gate is wired into both actual factory functions.
for filename in ['dflash_projection_row.py','dflash_projection_n/entry.py']:
    source=(O/'assembled'/prefix/filename).read_text();tree=ast.parse(source)
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='latency_gate' for n in ast.walk(tree))
    assert 'control_us' in source and 'rounds=12' in source
(O/'static_validation.json').write_text(json.dumps({'tests':tests,'variants':validation,'gate_wired_both_factories':True,'scope':'Source/application and injected latency-policy tests; not CUDA or multi-GPU execution'},indent=2))
print('PASS',len(tests),'admission scenarios;',len(validation),'patch variants')
