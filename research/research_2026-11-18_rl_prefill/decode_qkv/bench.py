"""Compare the pinned FA3 target entry against the fused packet on local CUDA."""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,json,statistics,sys,types
from pathlib import Path
from types import SimpleNamespace
import torch
import triton
import triton.language as tl
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SRC=ROOT/"sglang/python/sglang"
sys.path.insert(0,str(ROOT/"optimization_marlin_tiles_2026-11-15"))
from kernel_runtime import bootstrap

def extract(path, classname, names, bindings):
    nodes=ast.parse(path.read_text(encoding="utf-8-sig")).body
    if classname:
        nodes=next(n.body for n in nodes if isinstance(n,ast.ClassDef) and n.name==classname)
    keep=[n for n in nodes if getattr(n,"name",None) in names]
    assert len(keep)==len(names)
    mod=types.ModuleType("qkv_"+path.stem+"_"+(classname or "defs"))
    mod.__file__=str(path);mod.__dict__.update(bindings);sys.modules[mod.__name__]=mod
    exec(compile(ast.Module(ast.parse("from __future__ import annotations").body+keep,[]),str(path),"exec"),mod.__dict__)
    return mod

def qk(path):
    g=dict(torch=torch,triton=triton,tl=tl,FP8_E4M3_MAX=448.,is_arch_support_pdl=lambda:torch.cuda.get_device_capability()[0]>=9)
    k=extract(path,None,["_qk_rms_norm_fwd_kernel","_triton_config"],g)
    return extract(path,"FusedQKNorm",["_forward_triton"],k.__dict__)._forward_triton

def graph_fn(fn):
    for _ in range(3): fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        value=fn()
    return graph,value

def time_graph(graph,n=36):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n): graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)*1e3/n

def output(q,k,v,slots,qlen,window):
    # CUDA FP32 reference consumes the exact transient Q and physical KV after append.
    # The same page geometry, causal end alignment and 8:1 GQA as the FA3 caller.
    out=[]
    for j in range(qlen):
        end=slots.numel()-qlen+j+1
        start=max(0,end-window) if window else 0
        loc=slots[start:end]
        ku=k.view(torch.uint8)[loc].view(torch.float8_e4m3fn).view(-1,2,64).float()
        vu=v.view(torch.uint8)[loc].view(torch.float8_e4m3fn).view(-1,2,64).float()
        scores=torch.einsum("hgd,thd->hgt",q[j].float().reshape(2,8,64),ku)*(.125*5.4730064863875)
        out.append(torch.einsum("hgt,thd->hgd",scores.softmax(-1),vu))
    return torch.stack(out)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,required=True);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    bootstrap(ROOT/"sglang")
    from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache,store_cache
    assert can_use_store_cache(128,128)
    old=qk(HERE/"parent_muse_spark_qk_norm_rope_store.py");new=qk(HERE/"muse_spark_qk_norm_rope_store.py")
    prepare=extract(HERE/"parent_flashattention_backend.py","FlashAttentionBackend",["prepare_paged_mha_query"],dict(torch=torch)).prepare_paged_mha_query
    backend=SimpleNamespace(kv_cache_dtype_str="fp8_e4m3",kv_cache_is_mxfp8=False,kv_cache_dtype=torch.float8_e4m3fn,fa_impl_ver=3)
    layer=SimpleNamespace(head_dim=64,k_scale=None,v_scale=None)
    op=SimpleNamespace(_head_dim=64,_num_q_heads=16,_num_kv_heads=2,_eps=1.e-5)
    dev="cuda";fp8=torch.float8_e4m3fn;g=torch.Generator(device=dev).manual_seed(args.seed)
    lut=torch.arange(65536,device=dev,dtype=torch.int32).short().view(torch.bfloat16).to(fp8).view(torch.uint8)
    cap=80023
    posmax=80027
    inv=500000.**(-torch.arange(0,64,2,device=dev).float()/64)
    angles=torch.arange(posmax,device=dev).float()[:,None]*inv
    rotary=torch.cat((angles.cos(),angles.sin()),dim=1).contiguous()
    report=dict(scope="Scored FusedQKNorm Triton -> MHAPool dtype conversion -> native store_cache -> FA3 prepare_paged_mha_query, versus single fused entry. CUDAGraph; paired trials include all changed operations. H200 FA3 kernel and collective traffic excluded.",
                gpu=torch.cuda.get_device_name(),seed=args.seed,baseline_sha256=hashlib.sha256((ROOT/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((HERE/"candidate.diff").read_bytes()).hexdigest(),cases=[],reference=[])
    for batch,qlen,rope in [(1,1,False),(1,1,True),(4,1,False),(16,1,True),(32,1,False),(1,8,False),(1,8,True),(4,8,True),(16,8,False),(32,8,True)]:
        n=batch*qlen
        x=torch.empty(n,2304,device=dev,dtype=torch.bfloat16).normal_(generator=g)
        k0=torch.randint(0,115,(cap,128),device=dev,dtype=torch.uint8,generator=g).view(fp8)
        v0=torch.randint(0,115,(cap,128),device=dev,dtype=torch.uint8,generator=g).view(fp8)
        k1=k0.clone();v1=v0.clone()
        page=torch.randperm(cap-1,device=dev,generator=g)+1
        loc=page[-n:].clone();pos=torch.arange(75413,75413+n,device=dev,dtype=torch.int64)
        def base():
            q,k=old(op,x,pos if rope else None,rotary if rope else None)
            v=x[:,1152:1280]
            store_cache(k.to(fp8).view(torch.uint8),v.to(fp8).view(torch.uint8),k0.view(torch.uint8),v0.view(torch.uint8),loc,row_bytes=128,v_row_bytes=128,size_limit=cap)
            q=prepare(backend,q,None,None,layer,logical_batch_size=batch,kv_head_num=2,is_prefill=qlen>1)[0]
            return q,k
        def trial():
            q,k=new(op,x,pos if rope else None,rotary if rope else None,(k1,v1,loc,lut))
            q=prepare(backend,q,None,None,layer,logical_batch_size=batch,kv_head_num=2,is_prefill=qlen>1)[0]
            return q,k
        gb,rb=graph_fn(base);gc,rc=graph_fn(base);gt,rt=graph_fn(trial)
        times={k:[] for k in ("baseline","candidate","control")}
        for i in range(10):
            x.normal_(generator=g)
            sequence=[("baseline",gb),("candidate",gt),("control",gc)]
            for label,graph in (sequence if i%2==0 else sequence[::-1]): times[label].append(time_graph(graph))
        gb.replay();gt.replay()
        def errors():
            return [int((a.view(torch.uint8)!=b.view(torch.uint8)).sum()) for a,b in [(rb[0],rt[0]),(rb[1],rt[1]),(k0,k1),(v0,v1)]]
        errors_random=errors()
        if batch==1 and (qlen==1 and not rope or qlen==8 and rope):
            for length in [2048,74160]:
                slots=torch.cat((page[:length-qlen],loc))
                for window in [0,2048]:
                    a=output(rb[0],k0,v0,slots,qlen,window);b=output(rt[0],k1,v1,slots,qlen,window)
                    report["reference"].append(dict(tokens=length,q=qlen,window=window,byte_diffs=int((a.view(torch.uint8)!=b.view(torch.uint8)).sum())))
        # Alter graph replay addresses, padding, RoPE positions, and write patterns.
        loc.copy_(page[:n]);loc[0]=0;pos.add_(17);x.normal_(generator=g)
        x[:,1152:1280]=torch.tensor([448.,464.,466.,-464.,-466.,float("inf"),float("-inf"),-0.],device=dev).to(torch.bfloat16).repeat(16)
        x[0,:64]=float('inf'); x[0,1024:1088]=float('nan')
        gb.replay();gt.replay()
        adversarial=errors()
        row=dict(batch=batch,q=qlen,rope=rope,times=times,random=errors_random,changed_graph_overflow=adversarial)
        report["cases"].append(row)
        print(batch,qlen,rope,errors_random,adversarial,{k:round(statistics.median(v),2) for k,v in times.items()},flush=True)
        args.out.write_text(json.dumps(report,indent=2))
        assert errors_random==adversarial==[0,0,0,0] and all(x["byte_diffs"]==0 for x in report["reference"])
        del gb,gt,gc,k0,k1,v0,v1
if __name__=="__main__": main()

