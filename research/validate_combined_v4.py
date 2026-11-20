import importlib.util
import json
import sys
import types
from pathlib import Path
import torch

ROOT = Path(r'D:\OpenKernelChallenge\serving')
OPS = ROOT / 'sglang/python/sglang/kernels/ops/moe'
sys.path.insert(0, str(ROOT))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

utils = types.ModuleType('sglang.kernels.jit.utils')
utils.is_arch_support_pdl = lambda: False
sys.modules[utils.__name__] = utils
align = module('combined_align', OPS / 'moe_align_small_numel.py')
new = module('combined_finalize', OPS / 'muse_spark_post_expert_finalize_triton.py')
old = module('baseline_finalize', ROOT / 'iterations/before_combined_v4/muse_spark_post_expert_finalize_triton.py')
torch.manual_seed(604)
torch.cuda.init()

def run_align(ids, mapping, block):
    n = ids.numel()
    sorted_ids = torch.empty(n * block, device='cuda', dtype=torch.int32)
    experts = torch.empty(n, device='cuda', dtype=torch.int32)
    count = torch.empty(1, device='cuda', dtype=torch.int32)
    local = torch.empty_like(ids, dtype=torch.int32)
    align.moe_align_small_numel(ids, mapping.numel() + 1, block,
        sorted_ids, experts, count, expert_map=mapping, local_topk_ids=local)
    return local, sorted_ids, experts, count

align_cases = 0
for dtype in [torch.int32, torch.int64]:
    for n in [1, 2, 8, 16, 32, 63, 64]:
        for block in [8, 16, 32]:
            for mode in ['mixed', 'remote', 'local']:
                mapping = torch.full((256,), -1, device='cuda', dtype=torch.int32)
                mapping[64:96] = torch.arange(32, device='cuda', dtype=torch.int32)
                ids = torch.randint(-1, 256, (1, n), device='cuda', dtype=dtype)
                if mode == 'remote': ids.fill_(-1)
                if mode == 'local': ids.copy_(torch.randint(64, 96, (1,n), device='cuda', dtype=dtype))
                local, sorted_ids, experts, count = run_align(ids, mapping, block)
                expected = torch.where(ids < 0, -1, mapping[ids.clamp_min(0)]).to(torch.int32)
                assert torch.equal(local, expected)
                vals = expected.flatten().tolist()
                want_sorted, want_experts = [], []
                for e in sorted(set(vals)):
                    slots = [i for i, v in enumerate(vals) if v == e]
                    padded = ((len(slots) + block - 1) // block) * block
                    want_sorted.extend(slots + [n] * (padded - len(slots)))
                    want_experts.extend([e] * (padded // block))
                assert count.item() == len(want_sorted)
                assert sorted_ids[:len(want_sorted)].tolist() == want_sorted
                assert experts[:len(want_experts)].tolist() == want_experts
                align_cases += 1

final_cases = 0
max_abs = 0.0
max_oracle_error = 0.0
max_stock_oracle_error = 0.0
different_elements = 0
for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    for tokens in [1, 2, 8, 16, 32, 33]:
        for hidden in [1024, 4096, 8192]:
            for weight_first in [True, False]:
                x = torch.randn(tokens, 8, hidden, device='cuda', dtype=dtype)
                ids = torch.randint(-1, 3, (tokens, 8), device='cuda', dtype=torch.int32)
                ids[0].fill_(-1)
                x[ids < 0] = float('nan')
                scores = torch.rand(tokens, 8, device='cuda')
                w = torch.randn(hidden, device='cuda', dtype=torch.float32)
                if tokens == 2: w = None
                kwargs = dict(norm_weight=w, eps=1e-6, weight_first=weight_first, slot_expert_ids=ids)
                expected = old.fused_post_expert_norm_weighted_sum(x, scores, **kwargs)
                actual = new.fused_post_expert_norm_weighted_sum(x, scores, **kwargs)
                assert torch.isfinite(actual).all()
                delta = (actual.float() - expected.float()).abs().max().item()
                max_abs = max(max_abs, delta)
                different_elements += int((actual != expected).sum().item())
                f = x.double()
                if w is not None and weight_first: f = f * w.double()
                f = f * torch.rsqrt(f.square().mean(-1,keepdim=True) + 1e-6)
                if w is not None and not weight_first: f = f * w.double()
                ref = torch.where((ids >= 0).unsqueeze(-1), f * scores.double().unsqueeze(-1), 0).sum(1)
                max_oracle_error = max(max_oracle_error,(actual.double()-ref).abs().max().item())
                max_stock_oracle_error = max(max_stock_oracle_error,(expected.double()-ref).abs().max().item())
                atol = 2e-6
                rtol = 2e-6 if dtype == torch.float32 else torch.finfo(dtype).eps
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                final_cases += 1

# Nearby widths, different top-k arities, and the unmodified alignment API.
for topk in [2, 4, 16, 32]:
    x = torch.randn(2,topk,1536,device='cuda',dtype=torch.bfloat16)
    ids = torch.randint(-1,3,(2,topk),device='cuda',dtype=torch.int32)
    scores = torch.rand(2,topk,device='cuda')
    w = torch.randn(1536,device='cuda')
    actual = new.fused_post_expert_norm_weighted_sum(x,scores,w,slot_expert_ids=ids)
    expected = old.fused_post_expert_norm_weighted_sum(x,scores,w,slot_expert_ids=ids)
    torch.testing.assert_close(actual,expected,rtol=torch.finfo(x.dtype).eps,atol=2e-6)
    final_cases += 1
local = torch.tensor([[-1,0,3,0,2,-1,3,1]],device='cuda',dtype=torch.int32)
ordered = torch.empty(64,device='cuda',dtype=torch.int32)
experts = torch.empty(8,device='cuda',dtype=torch.int32)
count = torch.empty(2,device='cuda',dtype=torch.int32)
align.moe_align_small_numel(local,5,8,ordered,experts,count,write_inactive_prefix_count=True)
assert count.tolist() == [40,1]
assert experts[:5].tolist() == [-1,0,1,2,3]
align_cases += 1

# Replay on changing values and routing; no graph-capture-dependent state.
x = torch.randn(8,8,4096,device='cuda',dtype=torch.bfloat16)
ids = torch.randint(-1,3,(8,8),device='cuda',dtype=torch.int32)
scores = torch.rand(8,8,device='cuda')
w = torch.randn(4096,device='cuda')
for _ in range(3):
    new.fused_post_expert_norm_weighted_sum(x,scores,w,slot_expert_ids=ids)
mapping = torch.full((256,), -1, device='cuda', dtype=torch.int32)
mapping[64:96] = torch.arange(32,device='cuda',dtype=torch.int32)
global_ids = torch.randint(-1,256,(8,8),device='cuda',dtype=torch.int64)
run_align(global_ids,mapping,8)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = new.fused_post_expert_norm_weighted_sum(x,scores,w,slot_expert_ids=ids)
    aligned = run_align(global_ids,mapping,8)
for _ in range(8):
    x.normal_(); scores.uniform_(); ids.random_(-1,3); global_ids.random_(-1,256)
    graph.replay()
    reference = old.fused_post_expert_norm_weighted_sum(x,scores,w,slot_expert_ids=ids)
    torch.testing.assert_close(output, reference,rtol=0,atol=torch.finfo(x.dtype).eps/8)
    assert torch.equal(aligned[0],torch.where(global_ids<0,-1,mapping[global_ids.clamp_min(0)]).int())

def measure(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples=[]
    for _ in range(10):
        start.record(); g.replay(); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end)*1000/20)
    return sorted(samples)[len(samples)//2]

timings=[]
for tokens in [1,8,16,32]:
    x = torch.randn(tokens,8,4096,device='cuda',dtype=torch.bfloat16)
    ids = torch.randint(-1,3,(tokens,8),device='cuda',dtype=torch.int32)
    scores = torch.rand(tokens,8,device='cuda')
    kw = dict(norm_weight=w,slot_expert_ids=ids)
    baseline = measure(lambda: old.fused_post_expert_norm_weighted_sum(x,scores,**kw))
    candidate = measure(lambda: new.fused_post_expert_norm_weighted_sum(x,scores,**kw))
    timings.append(dict(tokens=tokens,baseline_us=baseline,candidate_us=candidate,ratio=baseline/candidate))

report = dict(alignment_cases=align_cases,finalize_cases=final_cases,max_absolute_difference=max_abs,
    differing_output_elements=different_elements, max_fp64_error=max_oracle_error, stock_max_fp64_error=max_stock_oracle_error,
    graph_changing_input_replays=8,device=torch.cuda.get_device_name(),timings=timings,
    limitation='Ampere laptop results; H200 PDL and full-model correctness/performance require judge check')
(ROOT/'validation/combined_v4.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
