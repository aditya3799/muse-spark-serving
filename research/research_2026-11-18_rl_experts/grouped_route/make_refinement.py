"""Construct extended grouped candidate without modifying the frozen discovery source."""
from pathlib import Path
here=Path(__file__).resolve().parent
src=(here/"grouped.py").read_text()
root=here.parents[1]
parent=(root/"release_2026-11-18/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py").read_text()
begin=parent.index("@triton.jit")
end=parent.index("# Two-phase")
kernel=parent[begin:end].replace("_post_expert_norm_weighted_sum_kernel(", "_mapped_finalize(").replace("    ids_ptr,","    ids_ptr,\n    ranks_ptr,").replace("base = (tok * TOPK + k) * H","base = tl.load(ranks_ptr + tok * TOPK + k) * H")
assert kernel.count("tl.load(ranks_ptr")==1
src=src.replace("def compute(layer,x,ids,split,variant):","def compute(layer,x,ids,split,variant,scores,norm):")
src=src.replace('variant=="split"', 'variant=="split_fused"')
src=src[:src.index("    slot_output=x.new_empty")]+"""    normalized=x.new_empty((m,h))
    _mapped_finalize[(m,)](down,norm,scores,normalized,ids,ranks,layer.w2_post_scale,
        HAS_SCALE=True,TOPK=8,H=h,eps=1e-8,APPLY_NORM=True,HAS_WEIGHT=True,WEIGHT_FIRST=True,
        HAS_IDS=True,ROUND_PRODUCT=False,BLOCK_H=h,num_warps=8,num_stages=2)
    return normalized
"""+kernel
(here/"grouped_fused.py").write_text(src)
bench=(here/"paired.py").read_text().replace('import grouped\n','import grouped_fused as grouped\n').replace('"grouped.py"','"grouped_fused.py"').replace('choices=("split","single_light","single_heavy","fused_finish"),default="split"','choices=("split_fused","single_light","single_heavy"),default="split_fused"').replace('args.variant=="split"','args.variant=="split_fused"')
bench=bench.replace("y=grouped.compute(layer,inner,ids,split,args.variant)","return env.after(grouped.compute(layer,inner,ids,split,args.variant,scores,norm))")
helper="""def finite_scales(e,n,h):
    v=torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8)
    codes=torch.tensor([i for i in range(256) if i not in (127,255)],device="cuda",dtype=torch.uint8)
    v[:,:254,0].copy_(codes)
    return v.view(torch.float8_e4m3fn)
"""
bench=bench.replace("exec(compile(weights,",helper+'weights=weights.replace(\'torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)\',"finite_scales(e,n,h)")\nexec(compile(weights,')
(here/"paired_fused.py").write_text(bench)
launch=(here/"launch.py").read_text().replace('("offline", "paired")','("offline", "paired", "paired_fused")')
(here/"launch_fused.py").write_text(launch)
