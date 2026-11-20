"""Canonical host-only Gloo receipts; CUDA helper and NCCL remain separately qualified."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;D=H/"decode_selector"
for stage in ("pilot","holdout"):
    r=json.loads((D/f"gloo_{stage}.json").read_text())
    c=dict(agent="speculation",candidate="packed_draft_topk_gloo_cpu",
        scope="8 live localhost Gloo CPU ranks, post-local-topk only, same rank-major all-gather layout and native global torch.topk; includes host eager bitpacking, explicitly excludes CUDA helpers and H200 NCCL",
        baseline_sha256="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b",
        variant_identity="eager host implementation, not the CUDA shipping candidate",active=True,complete=True,
        candidate_sha256=hashlib.sha256((H/'decode_gloo.py').read_bytes()).hexdigest(),seed=r["seed"],stage=stage,correct=all(v["wrongs"]==0 for v in r["shapes"].values()),
        bench=dict(instances=1,shapes={f"B{k}":dict(weight=1,times=v["times"]) for k,v in r["shapes"].items()}))
    (D/f"gloo_cpu_{stage}.json").write_text(json.dumps(c,indent=2))

