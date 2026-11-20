from pathlib import Path

source=Path('/mnt/c/Windows/Temp/build_projection_candidate.py').read_text()
source=source.replace("out = root / 'research_iteration_2026-11-15/projection_candidate'",
                      "out = root / 'research_iteration_2026-11-15/projection_fp32_candidate'")
old='''    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    n_local = weight.shape[0] // group.world_size
    start = group.rank_in_group * n_local
    local = bf16_gemm_dispatch(x, weight[start : start + n_local], None)
    return group.all_gather(local, dim=-1)'''
new='''    n_local = weight.shape[0] // group.world_size
    start = group.rank_in_group * n_local
    # A narrower output can select a reduced-precision cuBLAS reduction.
    # Request FP32 output explicitly, then round once to the stock BF16 dtype.
    local_fp32 = torch.mm(
        x, weight[start : start + n_local].t(), out_dtype=torch.float32
    )
    local = local_fp32.to(x.dtype)
    del local_fp32
    return group.all_gather(local, dim=-1)'''
assert source.count(old)==1
source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
