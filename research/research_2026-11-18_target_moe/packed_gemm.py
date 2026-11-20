"""GPU consumer for v139 K-permuted WGS weights, with no dense global weight copy in the kernel."""
import argparse
import ast
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from qualify_packing import DONOR, pack_stage


@triton.jit
def _project(X,W,Y,FP4,SCALE,BITS,M:tl.constexpr,N:tl.constexpr,GATE_UP:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,REGISTER_BITS:tl.constexpr=False):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    if GATE_UP:
        logical=cols%4096
        physical=(logical//8)*16+logical%8+(cols//4096)*8
    else:
        physical=cols
    unit=physical//256
    quad=physical%256//64
    warp=physical%64//16
    half=physical%16//8
    g=physical%8
    acc=tl.full((BM,BN),0,tl.float32)
    for block in range(4096//BK):
        ks=block*BK+tl.arange(0,BK)
        step=ks//16
        stage=step//16
        grp=step%16//4
        j=step%4
        t=ks%8//2
        q=ks%16//8
        low=ks%2
        base=((unit[None,:]*16+stage[:,None])*4+quad[None,:])*9216
        group=((grp[:,None]*4+warp[None,:])*8+g[None,:])
        words=base//4+((group*4+j[:,None])*4+t[:,None])
        word=tl.load(W.to(tl.pointer_type(tl.uint32))+words)
        scale_byte=tl.load(W+base+8192+(group*4+j[:,None])*2+half[None,:])
        register=q[:,None]*2+half[None,:]
        if REGISTER_BITS:
            # The first three register groups use three adjacent magnitude bits
            # and a sign nine positions above them. The fourth has its own permutation.
            shift=((6+3*register+low[:,None]*16)%32).to(tl.uint32)
            rotated=tl.inline_asm_elementwise("shf.r.wrap.b32 $0, $1, $1, $2;",constraints="=r,r,r",args=[word,shift],dtype=tl.uint32,is_pure=True,pack=1)
            regular=(rotated&7)|((rotated>>6)&8)
            halfword=word>>(low[:,None]*16).to(tl.uint32)
            fourth=((halfword>>4)&1)|((halfword&3)<<1)|(halfword&8)
            code=tl.where(register==3,fourth,regular)
        else:
            selector=register*8+low[:,None]*4
            code=tl.full((BK,BN),0,tl.uint32)
            for b in tl.static_range(4):
                shift=tl.load(BITS+selector+b).to(tl.uint32)
                code=code|(((word>>shift)&1)<<b)
        weight=(tl.load(FP4+code.to(tl.int32))*tl.load(SCALE+scale_byte.to(tl.int32))).to(tl.bfloat16)
        x=tl.load(X+rows[:,None]*4096+ks[None,:],mask=rows[:,None]<M,other=0)
        acc+=tl.dot(x,weight)
    tl.store(Y+rows[:,None]*N+cols[None,:],acc,mask=rows[:,None]<M)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    torch.manual_seed(925)
    names={"_wgs_copy","_pack_codes_nat","_stored_pos_mul"}
    nodes=[n for n in ast.parse(DONOR.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={"torch":torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(DONOR),"exec"),scope)
    positions=scope["_stored_pos_mul"]()
    bits=torch.tensor([positions[r,u,b] for r in range(4) for u in range(2) for b in range(4)],device="cuda",dtype=torch.int32)
    positive=[0,.5,1,1.5,2,3,4,6]
    fp4=torch.tensor(positive+[-v for v in positive],device="cuda")
    scale=torch.arange(256,device="cuda").byte().view(torch.float8_e4m3fn).float()
    receipts=[]
    for stage,n in (("w13",8192),("w2",4096)):
        codes=torch.randint(256,(1,n,2048),device="cuda",dtype=torch.uint8)
        for exact in (True,False):
            sf_bytes=torch.full((1,n,256),56,device="cuda",dtype=torch.uint8) if exact else torch.randint(24,70,(1,n,256),device="cuda",dtype=torch.uint8)
            packed=pack_stage(codes,sf_bytes.view(torch.float8_e4m3fn),stage,scope["_wgs_copy"],positions)
            # Independent dense oracle on sampled columns only; kernel must handle all columns.
            sample=torch.cat([torch.arange(0,n,79,device="cuda"),torch.tensor([7,8,15,16,255,256,4095,n-1],device="cuda")]).unique()
            chosen=codes[0,sample]
            nibbles=torch.stack([chosen%16,chosen//16],dim=-1).reshape(len(sample),4096)
            ref_weights=(fp4[nibbles.long()]*sf_bytes[0,sample].view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=1)).bfloat16().double()
            for m in (1,17,32):
                x=(torch.randint(-2,3,(m,4096),device="cuda").bfloat16() if exact else torch.randn((m,4096),device="cuda",dtype=torch.bfloat16))
                y=torch.empty((m,n),device="cuda",dtype=torch.float32)
                def project(register_bits=False):
                    _project[(triton.cdiv(m,16),n//32)](x,packed,y,fp4,scale,bits,m,n,stage=="w13",16,32,64,register_bits)
                project()
                old_output=y.clone()
                project(True)
                assert torch.equal(y,old_output), (stage,m,"register decode")
                expected=(x.double()@ref_weights.T).float()
                delta=(y[:,sample]-expected)
                assert torch.isfinite(y).all()
                rel=float(delta.norm()/expected.norm())
                if exact:
                    assert torch.equal(y[:,sample],expected), (stage,m,rel)
                else:
                    assert rel<2e-5,(stage,m,rel)
                receipts.append(dict(stage=stage,fixture="dyadic" if exact else "random_bf16",m=m,checked=int(expected.numel()),relative_l2=rel,max_abs=float(delta.abs().max())))
            # Perturb the canonical packed bytes and require the exact-reference case to reject it.
            if exact:
                original=y.clone()
                first=packed.view(-1)[:8192]
                first.bitwise_xor_(255)
                project()
                assert not torch.equal(original[:,sample],y[:,sample])
    data=dict(device=torch.cuda.get_device_name(),rows=receipts,negative_controls=2,
              scope="Full-K, unscaled W13/W2 single-expert GEMMs. Independent sampled FP64 oracle, production feature dimensions, canonical weight bytes. Does not execute donor WGMMA, routing, RMSNorm, collectives or serving.")
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/"results.json").write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)

if __name__=="__main__":
    main()

