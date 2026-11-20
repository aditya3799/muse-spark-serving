"""Measure full prefill shapes with bounded memory and verify changing graph inputs."""
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F

@torch.inference_mode()
def run():
    torch.manual_seed(815)
    k, n = 40960, 8192
    w = torch.empty((n,k), device="cuda", dtype=torch.bfloat16).normal_(std=k**-0.5)
    results = []
    for m in (4096,8192,16384):
        x = torch.empty((m,k), device="cuda", dtype=w.dtype).normal_()
        def full():
            return F.linear(x,w)
        def shard():
            return torch.mm(x, w[:n//8].T, out_dtype=torch.float32).to(x.dtype)
        def bench(fn):
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end)
        full()
        shard()
        base,local=[],[]
        for i in range(5):
            if i%2:
                local.append(bench(shard))
                base.append(bench(full))
            else:
                base.append(bench(full))
                local.append(bench(shard))
        live=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y=shard()
        extra=torch.cuda.max_memory_allocated()-live
        sample_rows = torch.tensor([0,m//2,m-1],device="cuda")
        sample_cols = torch.arange(1,n//8,23,device="cuda")
        oracle = (x[sample_rows].double() @ w[sample_cols].double().T).bfloat16().float()
        a=y[sample_rows][:,sample_cols].float()
        b=full()[sample_rows][:,sample_cols].float()
        results.append(dict(rows=m,full_ms=base,local_ms=local,median_ratio=statistics.median(f/s for f,s in zip(base,local)),
                            sampled_relative_to_oracle=float((a-oracle).norm()/oracle.norm()),
                            stock_sampled_relative_to_oracle=float((b-oracle).norm()/oracle.norm()),
                            shard_peak_extra_bytes=extra,gather_payload_bytes=m*n*2))
        del x,y
    x=torch.randn((128,k),device="cuda",dtype=w.dtype)
    sh=w[:n//8]
    def project():
        return torch.mm(x,sh.T,out_dtype=torch.float32).to(x.dtype)
    project()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured=project()
    changed=0
    for _ in range(5):
        x.normal_()
        graph.replay()
        assert torch.equal(captured,project())
        changed+=1
    print(json.dumps(dict(device=torch.cuda.get_device_name(),torch_version=torch.__version__,results=results,
                          changed_cuda_graph_replays=changed,scope="local projection only; excludes gather, RMSNorm, serving")))
    Path(__file__).with_name("prefill_graph.json").write_text(json.dumps(dict(results=results,changed_cuda_graph_replays=changed),indent=2))

run()
