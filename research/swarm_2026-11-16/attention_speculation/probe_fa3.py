import json
from pathlib import Path
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

out = Path(__file__).with_name('fa3_probe_result.json')
records = []
try:
    for batch in (1, 4):
        for query_len in (1, 4, 8):
            for window in ((-1, -1), (2047, 0)):
                torch.manual_seed(batch * 17 + query_len)
                length, hq, hk, dim, page = 4096, 16, 2, 64, 256
                q = torch.randn((batch * query_len, hq, dim), dtype=torch.bfloat16, device='cuda')
                k = torch.randn((batch * length // page, page, hk, dim), dtype=q.dtype, device='cuda')
                v = torch.randn_like(k)
                lens = torch.full((batch,), length, dtype=torch.int32, device='cuda')
                cuq = torch.arange(0, (batch + 1) * query_len, query_len, dtype=torch.int32, device='cuda')
                table = torch.arange(batch * length // page, dtype=torch.int32, device='cuda').view(batch, -1)
                kw = dict(q=q, k_cache=k, v_cache=v, page_table=table, cache_seqlens=lens, cu_seqlens_q=cuq, max_seqlen_q=query_len, causal=True, window_size=window)
                sched_kw = dict(batch_size=batch, max_seqlen_q=query_len, max_seqlen_k=length, num_heads=hq, num_heads_k=hk, headdim=dim, cache_seqlens=lens, qkv_dtype=q.dtype, cu_seqlens_q=cuq, page_size=page, causal=True, window_size=window)
                ref = flash_attn_with_kvcache(**kw)
                sched = get_scheduler_metadata(**sched_kw)
                got = flash_attn_with_kvcache(**kw, scheduler_metadata=sched)
                assert torch.equal(got, ref), (got-ref).abs().max().item()
                graph = torch.cuda.CUDAGraph()
                for _ in range(3):
                    flash_attn_with_kvcache(**kw, scheduler_metadata=sched)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    captured = flash_attn_with_kvcache(**kw, scheduler_metadata=sched)
                for changed_len in (17, 513, 2049, 4000):
                    lens.fill_(changed_len)
                    q.normal_()
                    sched.copy_(get_scheduler_metadata(**sched_kw))
                    graph.replay()
                    ref = flash_attn_with_kvcache(**kw)
                    assert torch.equal(captured, ref), (captured-ref).abs().max().item()
                records.append(dict(batch=batch, query_len=query_len, window=window, passed=True, scheduler_elements=sched.numel()))
                print(records[-1], flush=True)
    result = dict(passed=True, device=torch.cuda.get_device_name(), records=records)
except Exception as exc:
    result = dict(passed=False, exception_type=type(exc).__name__, error=str(exc), records=records)
    print(result, flush=True)
finally:
    out.write_text(json.dumps(result, indent=2))
