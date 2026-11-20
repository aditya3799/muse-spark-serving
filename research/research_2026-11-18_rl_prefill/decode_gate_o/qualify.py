"""Check the pinned call-site invariants and the scoped CUDA math layout without a GPU."""
import ast
import hashlib
import json
from pathlib import Path
import torch
from kernel import eligible
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
S=ROOT/"sglang/python/sglang"
def method(p,cls,k):
    tree=ast.parse((S/p).read_text(encoding="utf-8-sig"))
    return next(f for c in tree.body if getattr(c,"name",None)==cls for f in c.body if getattr(f,"name",None)==k)
f=method("srt/models/muse_spark.py","MuseSpark1xAttention","forward")
body=ast.unparse(f)
assert body.index("self.attn(q, k, v")<body.index("self._fused_o_norm_gate(output, o_gate)")<body.index("self.o_proj(output)")
row=ast.unparse(method("srt/layers/linear.py","RowParallelLinear","forward"))
assert "get_parallel().attn_tp_group.all_reduce(output_parallel)" in row and "use_symmetric_memory(get_parallel().attn_tp_group)" in row
boundary=ast.unparse(method("srt/models/muse_spark.py","MuseSpark1xDecoderLayer","forward"))
assert boundary.index("self.self_attn(")<boundary.index("self._fused_attn_boundary(")<boundary.index("self.mlp(")
gate=(S/"kernels/jit/csrc/muse_spark/norm_gate.cuh").read_text()
assert "fmaf(v.x, v.x, sum_of_squares)" in gate and "warp::reduce_sum<kThreadsPerRow>" in gate and "__fdividef(1.0f, 1.0f + __expf(-x))" in gate
# Keep the actual native config selector for all scored small-row counts.
def vec(m,heads,sm):
    want=16 if m*heads*16>sm*256 else 8
    candidates=[]
    for width in [4,8,16]:
        for warps in [1,2,4,8]:
            threads=32*warps; rowlanes=64//(width//2); rows=threads//rowlanes
            if heads%rows==0:candidates.append(((threads>=128,width==want,-abs(256-threads),width),width))
    return max(candidates)[1]
cases=[];torch.set_num_threads(2)
for sm in (20,132):
    for t in (1,4,8,13,16):
        gen=torch.Generator().manual_seed(7121+t)
        x=torch.randn((t,1024),generator=gen).bfloat16()
        packed=torch.randn((t,2304),generator=gen).bfloat16()
        g=packed[:,-1024:]
        assert vec(t,16,sm)==8 and eligible(x,g,torch.empty((8192,1024),dtype=x.dtype),64,sm)
        xv=x.float().view(t,16,16,4)
        acc=torch.zeros_like(xv[:,:,:,0])
        for j in range(4):acc=(xv[:,:,:,j].double().square()+acc.double()).float()
        for d in (8,4,2,1):
            perm=torch.arange(16)^d
            acc=acc+acc[:,:,perm]
        normalized=xv*torch.rsqrt(acc.unsqueeze(-1)/64+1.e-5)
        gated=(normalized.flatten(1)*torch.sigmoid(g.float())).bfloat16()
        independent=(x.float().view(t,16,64)*torch.rsqrt(x.float().view(t,16,64).square().mean(-1,keepdim=True)+1.e-5)).flatten(1)
        ref=(independent*torch.sigmoid(g.float())).bfloat16()
        gd=int((gated.view(torch.int16)!=ref.view(torch.int16)).sum())
        w=torch.randn((256,1024),generator=gen).bfloat16()
        # Simulated 16-at-a-time FP32 partial accumulation versus original CPU linear.
        tiled=torch.zeros((t,256))
        for ki in range(0,1024,16):tiled+=gated[:,ki:ki+16].float()@w[:,ki:ki+16].float().T
        y0=torch.nn.functional.linear(ref,w);y=tiled.bfloat16()
        # BF16 partials precede any rank reduction, so record every difference.
        cases.append(dict(sm=sm,t=t,gate_vec_bytes=8,gate_vs_unstructured_cpu=gd,partial_vs_cpu=int((y.view(torch.int16)!=y0.view(torch.int16)).sum())))
raw=dict(cases=cases,projected_shape=[8192,1024],slot_gate_stride=2304,source_checks=True,
         scope="CPU layout, pinned collective order and finite math controls. CPU software rsqrt/exp and tiled GEMM do not execute the true CUDA kernel; SM86/H200 parity still required.")
(H/"cpu.json").write_text(json.dumps(raw,indent=2));print(raw)

