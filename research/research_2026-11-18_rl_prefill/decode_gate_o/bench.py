"""Full-width correctness screen against pinned JIT gate and baseline BF16 torch linear."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from kernel import project
H=Path(__file__).resolve().parent
ROOT=H.parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(ROOT/"optimization_marlin_tiles_2026-11-15"))
    from kernel_runtime import bootstrap
    bootstrap(ROOT/"sglang")
    from sglang.kernels.ops.layernorm.muse_spark_norm_gate_jit import rms_norm_gate
    gen=torch.Generator(device="cuda").manual_seed(args.seed)
    def rand(shape):return torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=gen)
    def ms(g):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        for j in range(40):g.replay()
        b.record();b.synchronize();return a.elapsed_time(b)*1000/40
    def graph(fn):
        for _ in range(3):fn()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): y=fn()
        return g,y
    report=dict(seed=args.seed,scope="Pinned BF16 gate JIT plus complete unquantized O GEMM on one TP rank. Same-stream CUDA graphs, FP32 accumulation and BF16 partials. TP collective/downstream stages excluded.",correct=True,complete=False,rows={})
    for m in (1,4,8,13,16):
        w=rand((8192,1024))*.12;x=rand((m,1024));qkv=rand((m,2304));gate=qkv[:,-1024:];eps=1.e-5
        gate0=lambda:rms_norm_gate(x,gate,None,head_dim=64,eps=eps,has_norm=True)
        old=lambda:F.linear(gate0(),w)
        new=lambda:project(x,gate,w,eps)
        cases=[]
        for i in range(5):
            x.copy_(rand(x.shape));qkv.copy_(rand(qkv.shape))
            if i==1:x[:,::7]=0
            if i==2:w[:,1::2]=-w[:,::2];x[:,1::2]=x[:,::2];gate[:,1::2]=gate[:,::2]
            if i==3:x.mul_(2**-30);gate.mul_(40)
            if i==4:x.mul_(8);gate.mul_(40)
            ref_g=gate0();ref_y=F.linear(ref_g,w)
            y,g=project(x,gate,w,eps,True)
            gg=int((g.view(torch.int16)!=ref_g.view(torch.int16)).sum())
            erro=int((y.view(torch.int16)!=ref_y.view(torch.int16)).sum())
            isolated=F.linear(g,w)
            proj_diff=int((isolated.view(torch.int16)!=y.view(torch.int16)).sum())
            cases.append(dict(gate_bits_wrong=gg,rank_partial_bits_wrong=erro,projection_only_differences=proj_diff,
                              elements=y.numel(),max_abs=float((y.float()-ref_y.float()).abs().max())))
        row=report["rows"][m]=dict(correctness=cases)
        if any(c["gate_bits_wrong"] or c["rank_partial_bits_wrong"] for c in cases):
            report["correct"]=False
        else:
            graphs={a:graph(fn) for a,fn in (("baseline",old),("candidate",new),("control",old))}
            times={k:[] for k in graphs}
            for i in range(10):
                x.copy_(rand(x.shape));qkv.copy_(rand(qkv.shape))
                for k in (list(graphs) if i%2==0 else reversed(list(graphs))):times[k].append(ms(graphs[k][0]))
                if not torch.equal(graphs["baseline"][1].view(torch.int16),graphs["candidate"][1].view(torch.int16)):
                    report["correct"]=False;break
            row["times"]=times
        args.out.write_text(json.dumps(report,indent=2));print(m,cases[0],flush=True)
    report["complete"]=True;args.out.write_text(json.dumps(report,indent=2))
if __name__=="__main__":main()

