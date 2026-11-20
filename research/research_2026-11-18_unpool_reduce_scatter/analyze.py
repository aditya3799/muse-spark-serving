"""Quantify algorithmic traffic and sensitivity to BF16 transport reduction order."""
import json
from pathlib import Path

import torch

here = Path(__file__).resolve().parent
torch.set_num_threads(1)
weights = torch.randn(48,64, generator=torch.Generator().manual_seed(628)).to(torch.bfloat16)
def compare(a,b):
    return dict(unequal=int((a!=b).sum()), elements=a.numel(),
                relative_l2=float((a.float()-b.float()).norm()/a.float().norm()))

def sequential(parts):
    total=parts[0]
    for part in parts[1:]:
        total=total+part
    return total

def tree(parts):
    while len(parts)>1:
        parts=[parts[j]+parts[j+1] for j in range(0,len(parts),2)]
    return parts[0]

def project(rows):
    return torch.mm(rows.float(),weights.float().t()).to(torch.bfloat16)

order={}
for fixture in ("random","sparse","cancellation"):
    generator=torch.Generator().manual_seed(419)
    values=torch.randn(8,2048,64,generator=generator).to(torch.bfloat16)
    if fixture=="sparse":
        values *= (torch.rand(8,2048,1,generator=generator)>0.38)
    if fixture=="cancellation":
        for rank,scalar in enumerate([256,1,-256,1,256,1,-256,1]):
            values[rank,:,::2]=scalar
    parts=list(values.unbind())
    lhs,rhs=sequential(parts),tree(parts)
    order[fixture]={"shards":compare(lhs,rhs),"projected":compare(project(lhs),project(rhs))}
world, expert, hidden = 8,4096,8192
beta=(world-1)/world
traffic=[]
for rows in (512,753,2048,3012,4096,8624,16384):
    input_bytes=2*expert*rows
    output_bytes=2*hidden*rows
    eligible=rows>=512 and rows%world==0
    parent=beta*(2*input_bytes+output_bytes)
    modified=beta*(input_bytes+output_bytes)
    traffic.append(dict(rows=rows,eligible_by_shape=eligible,
        comparable_ring_parent_mib=parent/2**20 if eligible else None,
        candidate_mib=modified/2**20 if eligible else None,
        saving_mib=(parent-modified)/2**20 if eligible else 0))
report=dict(
    traffic=traffic,
    traffic_scope="Logical ring bytes sent per rank, not measured NVLink traffic; 753 and 3012 retain their original unsharded paths.",
    memory="For 16384 tokens the new BF16 row shard is 16 MiB; original full expert output is 128 MiB. In-place baseline all-reduce needs no new full output. Out-of-place baseline may have a different peak.",
    order_model="Eight-input synthetic CPU BF16 left-fold and balanced tree, summed in BF16 at each addition. This measures sensitivity; it does not identify the actual H200 NCCL protocol or its accumulator.",
    order=order)
(here/"bounds.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report["order"], indent=2))

