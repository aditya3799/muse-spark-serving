from pathlib import Path
import json
import torch
import torch.nn.functional as F
out=Path('/mnt/d/OpenKernelChallenge/serving/research_iteration_2026-11-15/projection_candidate')
rows=[]
def err(x,y):
    d=x.float()-y.float()
    return dict(relative_l2=(d.norm()/y.float().norm()).item(),max_abs=d.abs().max().item())
with torch.inference_mode():
    for m in (512,777):
        for k,n in ((8192,4096),(4096,8192)):
            torch.manual_seed(41+m)
            w=(torch.randn(n,k,device='cuda')/k**.5).to(torch.bfloat16)
            storage=torch.empty(m,k*2,device='cuda',dtype=torch.bfloat16)
            x=storage[:,::2]
            for seed in (73,101,307):
                torch.manual_seed(seed);storage.normal_()
                torch.backends.cuda.matmul.allow_tf32=False
                ref=F.linear(x.float(),w.float())
                for reduced in (False,True):
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
                    full=F.linear(x,w)
                    shards=torch.cat([F.linear(x,wi) for wi in w.chunk(8)],1)
                    strict=torch.cat([torch.mm(x,wi.t(),out_dtype=torch.float32).to(torch.bfloat16) for wi in w.chunk(8)],1)
                    row=dict(m=m,k=k,n=n,seed=seed,reduced=reduced,
                        sharded_vs_full=err(shards,full),full_vs_fp32=err(full,ref),
                        sharded_vs_fp32=err(shards,ref),fp32_output_sharded_vs_fp32=err(strict,ref),
                        fp32_output_sharded_vs_full=err(strict,full))
                    rows.append(row)
                    print(json.dumps(row),flush=True)
(out/'precision_diagnosis.json').write_text(json.dumps(dict(rows=rows,gpu=torch.cuda.get_device_name(),
    conclusion='Numerical diagnosis only. No thresholds changed; original failed screen remains recorded.'),indent=2))
