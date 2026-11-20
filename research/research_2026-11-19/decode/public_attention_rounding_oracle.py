"""CPU mechanism oracle: isolate missing BF16 rounding in donor Q/K prologue.

This does not execute or claim to emulate the full Hopper kernel. It holds
reduction and trig fixed to isolate the difference explicitly visible in source.
"""
import hashlib,json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;torch.manual_seed(617849)
rows=8192;d=64
x=torch.randn(rows,d).bfloat16().float()
normalized=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)
pos=torch.randint(1,130000,(rows,))
phase=pos[:,None].float()*500000.**(-torch.arange(0,d,2).float()/d)
cs,sn=phase.cos(),phase.sin()
def rotate(z):
 a,b=z[:,0::2],z[:,1::2]
 return torch.stack((a*cs-b*sn,b*cs+a*sn),-1).flatten(-2)
records=[]
for rope in (False,True):
 parent=normalized
 if rope:parent=rotate(parent.bfloat16().float())
 parent=parent.bfloat16().to(torch.float8_e4m3fn)
 donor=rotate(normalized) if rope else normalized
 donor=donor.to(torch.float8_e4m3fn)
 bad=(parent.view(torch.uint8)!=donor.view(torch.uint8))
 records.append(dict(rope=rope,changed_fp8_codes=int(bad.sum()),total=bad.numel(),fraction=float(bad.float().mean()),max_abs=float((parent.float()-donor.float()).abs().max())))
doc=dict(scope='CPU rounding-path mechanism only. Same FP32 norm and trig for both; omits donor reduction/trig differences, no GPU timing.',seed=617849,
 parent='pinned FusedQKNorm casts normalized Q/K to BF16 before RoPE, then BF16 before FP8',
 donor='alygg77 6cae49a6 attention prologue passes FP32 normalized/rotated Q/K directly to FP8',
 rows=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/'public_attention_rounding_oracle.json').write_text(json.dumps(doc,indent=2));print(json.dumps(doc,indent=2))
