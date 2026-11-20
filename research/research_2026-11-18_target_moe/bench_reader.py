"""Paired local GEMM timings with equal inputs and the same FP32 output boundary."""
import argparse
import ast
import json
import random
import statistics
from pathlib import Path

import torch
import triton
from bench_experts import paired_ci
from packed_gemm import _project, DONOR, pack_stage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    names={"_wgs_copy","_pack_codes_nat","_stored_pos_mul"}
    nodes=[node for node in ast.parse(DONOR.read_text()).body if getattr(node,"name",None) in names]
    ns={"torch":torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(DONOR),"exec"),ns)
    positions=ns["_stored_pos_mul"]()
    bits=torch.tensor([positions[r,u,b] for r in range(4) for u in range(2) for b in range(4)],device="cuda",dtype=torch.int32)
    pos=[0,.5,1,1.5,2,3,4,6]
    fp4=torch.tensor(pos+[-v for v in pos],device="cuda")
    scale=torch.arange(256,device="cuda").byte().view(torch.float8_e4m3fn).float()
    torch.manual_seed(651)
    rows=[]
    for stage,n in (("w13",8192),("w2",4096)):
        codes=torch.randint(256,(1,n,2048),device="cuda",dtype=torch.uint8)
        sf=torch.randint(24,70,(1,n,256),device="cuda",dtype=torch.uint8)
        packed=pack_stage(codes,sf.view(torch.float8_e4m3fn),stage,ns["_wgs_copy"],positions)
        dense=torch.empty((n,4096),device="cuda",dtype=torch.bfloat16)
        for begin in range(0,n,256):
            chunk=codes[0,begin:begin+256]
            ids=torch.stack([chunk%16,chunk//16],dim=-1).reshape(-1,4096)
            dense[begin:begin+256]=(fp4[ids.long()]*scale[sf[0,begin:begin+256].long()].repeat_interleave(16,dim=1)).bfloat16()
        for m in (8,32,128):
            x=torch.randn((m,4096),device="cuda",dtype=torch.bfloat16)
            y=torch.empty((m,n),device="cuda",dtype=torch.float32)
            def run(variant):
                if variant==2:
                    return torch.mm(x,dense.T,out_dtype=torch.float32)
                _project[(triton.cdiv(m,16),n//32)](x,packed,y,fp4,scale,bits,m,n,stage=="w13",16,32,64,variant==1)
                return y
            assert torch.equal(run(0).clone(),run(1))
            for v in range(3):
                run(v)
            torch.cuda.synchronize()
            samples=[]
            rng=random.Random(78)
            for i in range(15):
                order=[0,1,2]
                rng.shuffle(order)
                x.normal_()
                t=[0]*3
                for v in order:
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record()
                    run(v)
                    b.record()
                    b.synchronize()
                    t[v]=a.elapsed_time(b)*1000
                samples.append(t)
            med=list(map(statistics.median,zip(*samples)))
            rows.append(dict(stage=stage,m=m,table_us=med[0],register_us=med[1],dense_us=med[2],
                             decode_ratio_ci=paired_ci([(a,b) for a,b,c in samples]),samples_us=samples))
            print(stage,m,med,flush=True)
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/"results.json").write_text(json.dumps(dict(device=torch.cuda.get_device_name(),rows=rows,
        scope="Component diagnostic. Dense BF16 GPU weights pre-exist only in the comparator. The candidate reads packed weights directly. No routing, H200 or serving timing."),indent=2))

if __name__=="__main__":
    main()

