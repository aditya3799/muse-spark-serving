import json, statistics
from pathlib import Path
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

torch.manual_seed(191)
rows = []
for batch, length in ((1, 4096), (4, 4096), (1, 32768), (4, 32768)):
    qlen, hq, hk, dim, page, layers = 8, 16, 2, 64, 256, 32
    q = torch.randn(batch * qlen, hq, dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch * length // page, page, hk, dim, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    lens = torch.full((batch,), length, device='cuda', dtype=torch.int32)
    cuq = torch.arange(0, (batch + 1) * qlen, qlen, device='cuda', dtype=torch.int32)
    table = torch.arange(batch * length // page, device='cuda', dtype=torch.int32).view(batch, -1)
    kw = dict(q=q, k_cache=k, v_cache=v, page_table=table, cache_seqlens=lens, cu_seqlens_q=cuq, max_seqlen_q=qlen, causal=True)
    skw = dict(batch_size=batch, max_seqlen_q=qlen, max_seqlen_k=length, num_heads=hq, num_heads_k=hk, headdim=dim, cache_seqlens=lens, qkv_dtype=q.dtype, cu_seqlens_q=cuq, page_size=page, causal=True)
    graphs = []
    for reuse in (False, True):
        for _ in range(3):
            flash_attn_with_kvcache(**kw)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sched = get_scheduler_metadata(**skw) if reuse else None
            outputs = [flash_attn_with_kvcache(**kw, scheduler_metadata=sched) for _ in range(layers)]
        graphs.append(graph)
    vals = [[], []]
    for sample in range(10):
        for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10): graphs[index].replay()
            end.record(); end.synchronize()
            vals[index].append(start.elapsed_time(end) / 10)
    rows.append(dict(batch=batch, context=length, layers=layers, baseline_ms=statistics.median(vals[0]), reuse_ms=statistics.median(vals[1]), ratio=statistics.median(vals[0])/statistics.median(vals[1]), samples=vals))
    print({k:v for k,v in rows[-1].items() if k != 'samples'}, flush=True)
    del graphs, outputs, q, k, v
Path(__file__).with_name('fa3_bench_result.json').write_text(json.dumps(dict(device=torch.cuda.get_device_name(), note='Synthetic BF16 attention-only chain, not serving or Hopper FP8', rows=rows), indent=2))
