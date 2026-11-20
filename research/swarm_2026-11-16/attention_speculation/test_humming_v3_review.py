"""CPU-only exact-method dispatch/guard and arithmetic-tail source review."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / 'swarm_2026-11-16/moe/candidate_v3/serving_export/python/sglang'
fp = EXPORT / 'srt/layers/muse_spark/muse_spark_fp4_moe.py'
fin = EXPORT / 'kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
base = ROOT / 'sglang/python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
manifest = json.loads((EXPORT.parents[2] / 'manifest.json').read_text(encoding='utf-8'))
for rel, expected in manifest['sources'].items():
    path = EXPORT.parents[1] / rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

def extract(path, name):
    node = next(n for n in ast.walk(ast.parse(path.read_text(encoding='utf-8'))) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    for n in ast.walk(node):
        if isinstance(n, ast.arg): n.annotation = None
        if isinstance(n, ast.FunctionDef): n.returns = None
    return node

def run_def(node, ns):
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<exact-source-method>', 'exec'), ns)
    return ns[node.name]

calls = []
def fused(*args, **kw): calls.append(('fused', kw)); return 'fused-result'
def generic(**kw): calls.append(('generic', kw)); return 'generic-result'
forward = run_def(extract(fp, '_forward_marlin'), {'fused_post_expert_norm_weighted_sum': fused, 'finalize_identity_with_post_expert_norm': generic})
recognized, arbitrary = NS(recognized=True), NS(recognized=False)
route_cases = 0
for humming in (False, True):
    for old, new in ((recognized,recognized),(arbitrary,recognized),(recognized,arbitrary),(arbitrary,arbitrary)):
        self = NS(use_humming=humming, _post_expert_norm=old, _post_expert_norm_weight='weight' if old.recognized else None,
                  _post_expert_norm_eps=1e-8, _post_expert_norm_weight_first=True,
                  _humming_layer=NS(w2_post_scale='scale'))
        def setter(norm):
            calls.append(('setter', norm.recognized))
            self._post_expert_norm = norm
            self._post_expert_norm_weight = 'weight' if norm.recognized else None
        self.set_post_expert_norm = setter
        def project(*args, **kw): calls.append(('project', kw)); return ('outputs','ids')
        self._marlin_expert_outputs = project
        calls.clear()
        got = forward(self, NS(dtype='bf16'), 'scores', 'ids', new)
        defer = humming and new.recognized
        assert next(v for k,v in calls if k=='project')['defer_downscale'] == defer
        assert got == ('fused-result' if new.recognized else 'generic-result')
        if new.recognized:
            assert calls[-1][1]['expert_output_scale'] == ('scale' if defer else None)
        if humming and old is not new:
            assert calls[0][0] == 'setter'
        route_cases += 1

class Tensor:
    def __init__(self, shape, dtype='bf16', device='cuda', contiguous=True):
        self.shape=tuple(shape); self.dtype=dtype; self.device=device
        self.is_cuda=device.startswith('cuda'); self.ndim=len(shape); self._contiguous=contiguous
    def numel(self): return math.prod(self.shape)
    def is_contiguous(self): return self._contiguous
    def contiguous(self): return self

class Kernel:
    def __init__(self,name): self.name=name
    def __getitem__(self,grid):
        def launch(*args, **kw): calls.append((self.name,grid,args,kw))
        return launch

kernel_names = ['_post_expert_norm_weighted_sum_kernel','_post_expert_inv_rms_kernel','_post_expert_tiled_combine_kernel']
ns = {name: Kernel(name) for name in kernel_names}
ns.update(torch=NS(bfloat16='bf16',int32='i32',int64='i64',float32='f32',empty=lambda shape,**kw: Tensor(shape,**kw)),
          triton=NS(next_power_of_2=lambda x:1<<(x-1).bit_length(),cdiv=lambda a,b:(a+b-1)//b), _MIN_TOKENS_FOR_FUSED=32)
finalize = run_def(extract(fin,'fused_post_expert_norm_weighted_sum'),ns)
dispatch_cases = 0
for m in (0,1,2,8,9,32):
    for scale_enabled in (False,True):
        for apply_norm in (False,True):
            for round_product in (False,True):
                calls.clear()
                scale = Tensor((32,)) if scale_enabled else None
                result=finalize(Tensor((m,8,4096)),Tensor((m,8),'f32'),norm_weight=Tensor((4096,)),slot_expert_ids=Tensor((m,8),'i32'),expert_output_scale=scale,apply_norm=apply_norm,round_product=round_product)
                assert result.shape==(m,4096)
                expected=0 if m==0 else (2 if 2<=m<=8 and apply_norm and not round_product else 1)
                assert len(calls)==expected
                for _,_,args,kw in calls:
                    assert kw['HAS_SCALE']==scale_enabled
                    if scale_enabled: assert any(v is scale for v in args)
                dispatch_cases+=1

guard_cases=0
invalid = [dict(slot_expert_ids=None),dict(expert_output_scale=Tensor((32,),'f32')),
           dict(expert_output_scale=Tensor((1,32))),dict(expert_output_scale=Tensor((0,))),
           dict(expert_output_scale=Tensor((32,),contiguous=False)),dict(slot_expert_ids=Tensor((2,8),'f32')),
           dict(slot_expert_ids=Tensor((1,8),'i32')),dict(slot_expert_ids=Tensor((2,8),'i32','cuda:1')),
           dict(expert_output=Tensor((2,8,4096),'f32')),dict(expert_output=Tensor((2,8,4096),device='cpu'))]
for change in invalid:
    args=dict(expert_output=Tensor((2,8,4096)),topk_scores=Tensor((2,8),'f32'),slot_expert_ids=Tensor((2,8),'i32'),expert_output_scale=Tensor((32,)))
    args.update(change)
    try: finalize(**args)
    except (TypeError,ValueError): guard_cases+=1
    else: raise AssertionError('Guard did not reject '+repr(change))

tails=0
rounding = ast.parse('x = (x * scale).to(tl.bfloat16).to(tl.float32)').body[0]
for name in kernel_names:
    new=extract(fin,name); old=extract(base,name)
    scale_if = next(n for n in ast.walk(new) if isinstance(n,ast.If) and ast.dump(n.test)==ast.dump(ast.Name(id='HAS_SCALE',ctx=ast.Load())))
    assert ast.dump(scale_if.body[-1])==ast.dump(rounding)
    # The inserted scaling runs inside the existing live-route branch.
    live = next(n for n in ast.walk(new) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='live')
    oldlive = next(n for n in ast.walk(old) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='live')
    assert scale_if in live.body
    idx=live.body.index(scale_if)
    # Every operation following scale rounding equals the old norm/combine tail.
    assert [ast.dump(n) for n in live.body[idx+1:]] == [ast.dump(n) for n in oldlive.body[idx:]]
    tails+=1

report={'status':'PASS','scope':'CPU exact-method dispatch/guards and AST tails only; no Triton compilation/numerical/graph test',
        'source_hashes':manifest['sources'],'forward_route_cases':route_cases,'finalizer_dispatch_cases':dispatch_cases,
        'rejected_guard_cases':guard_cases,'unchanged_arithmetic_tails':tails}
out=Path(__file__).with_name('humming_v3_cpu_review.json')
out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
