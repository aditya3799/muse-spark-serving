"""Small synthetic geometry exercising the actual exported adapter code under sanitizers.
No performance claim. This supplements, not replaces, the real-weight tests.
"""
import argparse,hashlib,json,pathlib
import torch
O=pathlib.Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-20/local_validation')
source=O/'adapter_probe.py';text=source.read_text()
old="out=R/'research_2026-11-20'/'local_validation'/f'adapter_probe_{seed}_{rows[0]}.json'"
assert text.count(old)==1
text=text.replace(old,"out=R/'research_2026-11-20'/'local_validation'/'sanitizer'/f'adapter_probe_{seed}_{rows[0]}.json'")
text=text.replace('40960','1024')
ns={'__name__':'sanitizer_adapter','__file__':str(source)}
exec(compile(text,str(source)+'[1024 fixture]','exec'),ns)
def fixture():
    gen=torch.Generator(device='cuda').manual_seed(20112083)
    w=torch.randn((1024,1024),device='cuda',dtype=torch.bfloat16,generator=gen)
    return (w,None,None,None),{'synthetic':True,'shape':[1024,1024],'original_harness_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'purpose':__doc__}
ns['real_row_chain'].checkpoint=fixture
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    (O/'sanitizer').mkdir(exist_ok=True)
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):ns['main'](a.seed,[a.rows])
    result=json.loads((O/'sanitizer'/f'adapter_probe_{a.seed}_{a.rows}.json').read_text())
    assert result['complete'] and result['correct']
    assert all(c['enabled'] and c.get('changed_graph_exact') for c in result['cases']),result
