"""Compare stock BF16 and FP32-output shards on identical inputs."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

@torch.inference_mode()
def run():
    torch.manual_seed(423)
    w = torch.randn((8192, 40960), device="cuda", dtype=torch.bfloat16) * (40960 ** -0.5)
    results = []
    for m in (1, 8, 32, 128, 256, 753, 1024):
        x = torch.randn((m,40960), device="cuda", dtype=w.dtype)
        ref = F.linear(x,w)
        sampled = torch.arange(7,8192,191,device="cuda")
        oracle = (x.double() @ w[sampled].double().T).bfloat16().float()
        variants = {
            "stock_bf16":lambda ws: F.linear(x,ws),
            "fp32_output":lambda ws: torch.mm(x,ws.T,out_dtype=torch.float32).bfloat16(),
        }
        for name,fn in variants.items():
            test = torch.cat([fn(ws) for ws in w.chunk(8)],dim=1)
            d = test.float()-ref.float()
            results.append(dict(rows=m, variant=name, exact=float((d==0).float().mean()),
                                relative_l2=float(d.norm()/ref.float().norm()),
                                reference_relative_l2=float((test[:,sampled].float()-oracle).norm()/oracle.norm())))
    print(json.dumps(results))
    Path(__file__).with_name("precision_comparison.json").write_text(json.dumps(results,indent=2))

run()
