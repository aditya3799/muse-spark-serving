"""Compare output-feature and token-row partitions; model only local gather layout work."""
import json
from pathlib import Path
from statistics import median
import torch
from torch.nn import functional as F

def timing(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)

@torch.inference_mode()
def main():
    torch.manual_seed(918)
    tp,k,n=8,40960,8192
    w=torch.empty((n,k),device="cuda",dtype=torch.bfloat16).normal_(std=k**-0.5)
    results=[]
    for m in (8,32,128,256,753,1024,4096,16384):
        rows=(m+tp-1)//tp
        x=torch.randn((rows*tp,k),device="cuda",dtype=w.dtype)
        def row():
            return F.linear(x[:rows],w)
        def column():
            return torch.mm(x[:m],w[:n//tp].T,out_dtype=torch.float32).bfloat16()
        # Materialize the same rank-major order as the pinned group coordinator.
        raw=torch.empty((tp,m,n//tp),device="cuda",dtype=w.dtype)
        def reorder():
            return raw.movedim(0,1).reshape(m,n)
        row()
        column()
        reorder()
        stats=[[],[],[]]
        ops=[row,column,reorder]
        for i in range(5):
            for j in ((0,1,2) if i%2==0 else (2,1,0)):
                stats[j].append(timing(ops[j]))
        # Compare the union of every live row across eight simulated ranks.
        # Chunk the comparison to keep the full prefill case within 6 GiB.
        ref=F.linear(x[:m],w)
        exact=0
        error_sq=ref_sq=0.
        for r in range(tp):
            start,end=rows*r,min(rows*(r+1),m)
            if start>=end:
                continue
            value=F.linear(x[start:start+rows],w)[:end-start]
            target=ref[start:end]
            exact+=(value==target).sum().item()
            error_sq+=(value.float()-target.float()).square().sum().item()
            ref_sq+=target.float().square().sum().item()
        results.append(dict(rows=m,local_rows=rows,token_median_ms=median(stats[0]),feature_median_ms=median(stats[1]),
                          feature_reorder_ms=median(stats[2]),token_exact_fraction=exact/(m*n),
                          token_relative_l2=(error_sq/ref_sq)**0.5,raw_ms=stats))
        del ref, x, raw, target, value
    output=dict(device=torch.cuda.get_device_name(),results=results,
                scope="replicated original BF16 weights for row shards; single GPU; no interconnect, serving or RMSNorm timing")
    Path(__file__).with_name("axis_comparison.json").write_text(json.dumps(output,indent=2))
    for result in results:
        print({k:v for k,v in result.items() if k!="raw_ms"})
main()
