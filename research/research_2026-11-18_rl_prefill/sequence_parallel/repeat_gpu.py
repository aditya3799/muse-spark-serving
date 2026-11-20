"""Check the original rank position, gathered order, and repeated rows before both router reductions."""
import ast
import faulthandler
import json
from typing import Optional
import torch
from torch import nn
import triton
import triton.language as tl
from router_gpu import source,HERE,ROOT

def main():
    faulthandler.dump_traceback_later(40,repeat=True)
    print("starting independent native geometry comparison",flush=True)
    path=ROOT/"sglang/python/sglang/kernels/ops/moe/muse_spark_routing.py"
    funcs=[f.name for f in ast.parse(path.read_bytes()).body if isinstance(f,ast.FunctionDef) and f.name!="fused_muse_spark_routing"]
    ops=source(path,dict(torch=torch,triton=triton,tl=tl,Optional=Optional,register_custom_op=lambda **kw:lambda f:f),funcs)
    model=source(HERE/"muse_spark.py",dict(torch=torch,nn=nn),["MuseSpark1xRouter"])
    def top8(parts,bias):
        high,low=parts.chunk(2,dim=-1);logits=high+low
        w=torch.empty(logits.shape[0],8,device="cuda");ids=torch.empty_like(w,dtype=torch.int32);packed=torch.empty_like(ids)
        ops.fused_muse_spark_routing_op(logits,bias,w,ids,packed,8,1.,1.,1e-15,0,False,True)
        return w,ids,packed
    def mismatch(x,y):
        return int(torch.count_nonzero(x.view(torch.int32)!=y.view(torch.int32)))
    data=[]
    for seed in (402781,708611):
        gen=torch.Generator(device="cuda").manual_seed(seed)
        for style in ("random","near_tie","cancellation"):
            m,k,n=16384,8192,256
            x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16,generator=gen)
            router=model.MuseSpark1xRouter(k,n,torch.float32).cuda()
            w=torch.randn(n,k,device="cuda",generator=gen)/k**.5
            if style=="near_tie":w[1::2]=w[::2]+torch.randn(n//2,k,device="cuda",generator=gen)*2e-10
            if style=="cancellation":
                w[:,1::2]=-w[:,::2];x[:,1::2]=x[:,::2];x[:,-1]+=0.00390625
            router.weight.copy_(w);router._build_split2();packed=router._split2_weight
            baseline=torch.mm(x,packed,out_dtype=torch.float32)
            bref=top8(baseline,router.e_score_correction_bias)
            gathered=x.chunk(8,dim=0);original=torch.cat(gathered,dim=0)
            whole=torch.mm(original,packed,out_dtype=torch.float32)
            item={"seed":seed,"style":style,"global_rows":m,"gathered_component_mismatch":mismatch(whole,baseline),"rank":[]}
            for rank,piece in enumerate(gathered):
                idx=slice(rank*(m//8),(rank+1)*(m//8))
                full=piece.repeat((8,1))
                local=torch.mm(full,packed,out_dtype=torch.float32)[idx]
                actual=top8(local,router.e_score_correction_bias)
                item["rank"].append({"rank":rank,"fp32_components":mismatch(local,baseline[idx]),
                                     "top8_weights":mismatch(actual[0],bref[0][idx]),"top8_ids":mismatch(actual[1],bref[1][idx]),
                                     "top8_packed":mismatch(actual[2],bref[2][idx])})
            if style=="random":
                rank=3;sl=slice(rank*(m//8),(rank+1)*(m//8));piece=gathered[rank]
                times={"baseline":[],"candidate":[],"control":[]}
                for i in range(6):
                    for arm in (("baseline","candidate","control") if i%2==0 else ("control","candidate","baseline")):
                        st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True)
                        st.record()
                        inp=piece.repeat((8,1)) if arm=="candidate" else x
                        parts=torch.mm(inp,packed,out_dtype=torch.float32)
                        top8(parts[sl] if arm=="candidate" else parts,router.e_score_correction_bias)
                        en.record();en.synchronize()
                        times[arm].append(st.elapsed_time(en)*1000)
                item["rank3_local_us_excluding_peer_transport"]=times
            data.append(item)
            (HERE/"repeat_gpu.json").write_text(json.dumps(data,indent=2))
            print(seed,style,item["gathered_component_mismatch"],[r["fp32_components"] for r in item["rank"]],flush=True)
if __name__=="__main__":main()

