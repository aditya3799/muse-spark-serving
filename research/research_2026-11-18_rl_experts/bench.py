"""Paired complete routed-expert comparison on the frozen scored parent."""
import argparse,ast,hashlib,inspect,json,statistics,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"research_2026-11-18_target_moe"))
import bench_experts as original
import triton
import triton.language as tl

@triton.jit
def _fuse(X,IDS,SCALE,Y,D:tl.constexpr,TILE:tl.constexpr):
    row=tl.program_id(0)
    expert=tl.load(IDS+row)
    if expert>=0:
        j=tl.program_id(1)*TILE+tl.arange(0,TILE)
        scale=tl.load(SCALE+expert).to(tl.float32)
        gate=(tl.load(X+row*D*2+j).to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
        up=(tl.load(X+row*D*2+D+j).to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
        # Equivalent activation flags in pinned native CUDA use --use_fast_math.
        from_exp=tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;",constraints="=f,f",args=[-gate*1.4426950408889634],dtype=tl.float32,is_pure=True,pack=1)
        gate_act=tl.inline_asm_elementwise("div.approx.ftz.f32 $0, $1, $2;",constraints="=f,f,f",args=[gate,1.+from_exp],dtype=tl.float32,is_pure=True,pack=1)
        tl.store(Y+row*D+j,gate_act*up)

def scale_silu(x,y,ids,scales):
    tile=1024
    _fuse[(ids.numel(),y.shape[1]//tile)](x,ids,scales,y,y.shape[1],tile,num_warps=4)

def adapt(fn, boundary):
    source=inspect.getsource(fn)
    lines=[]
    for line in source.splitlines():
        if "scale_active_rows(cache1," in line:
            continue
        if "silu_and_mul(cache1," in line:
            line="    scale_silu(cache1, cache2, local_ids, layer.w13_post_scale)"
        lines.append(line)
    result=dict(fn.__globals__,scale_silu=boundary)
    exec(compile("\n".join(lines),fn.__code__.co_filename+":paired","exec"),result)
    return result[fn.__name__]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--serial-full",action="store_true")
    parser.add_argument("--envelope",choices=["experts","ffn"],default="experts")
    parser.add_argument("--rows",type=int,nargs="+",default=[8,32,128,256,753,16384])
    parser.add_argument("--reps",type=int,default=4)
    parser.add_argument("--patterns",nargs="+",default=["uniform","cluster"])
    parser.add_argument("--candidate",choices=["scale_silu","route_ballot","compact_live"],default="scale_silu")
    args=parser.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    for p in (original.OLD/"humming_vendor",original.OLD/"phase2",ROOT/"optimization_marlin_tiles_2026-11-15"):
        sys.path.insert(0,str(p))
    from kernel_runtime import bootstrap
    bootstrap(ROOT/"sglang")
    import torch
    from fixture_tuning_cache import cached_threshold_for_preparation
    moe=original.SOURCE/"kernels/ops/moe"
    original.module("sglang.kernels.ops.moe.muse_spark_humming_scale",moe/"muse_spark_humming_scale.py")
    split=original.module("sglang.kernels.ops.moe.muse_spark_humming_split",moe/"muse_spark_humming_split.py")
    finish=original.module("scored_finalize",moe/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
    path=original.SOURCE/"srt/layers/muse_spark/muse_spark_nvfp4_moe.py"
    tree=ast.parse(path.read_text()); names=("_prepare_humming_experts","_humming_expert_outputs_default","_humming_expert_outputs")
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={"torch":torch};exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),"exec"),ns)
    digest=hashlib.sha256((original.RELEASE/"candidate.diff").read_bytes()).hexdigest()
    assert digest=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    report=dict(agent="experts",candidate=args.candidate,scope="Humming W13 + W2 and scored BF16 post-norm/weighted sum, RTX3050 SM86, 32 rank-local of 256 experts, H=I=4096, top8",baseline_sha256=digest,seed=args.seed,correct=True,active=True,complete=False,bench=dict(instances=1,shapes={}))
    if args.envelope=="ffn":
        report["scope"]="Rank-local TP8-shaped BF16 pooler 8192?4096; pinned GPU pre-expert weighted RMSNorm, complete Humming and score post-expert norm/combine, local unpooler 4096?8192; synthetic external ids/scores. Includes no attention, router GEMM, cross-rank bandwidth/waiting, or post-FFN residual. EP sum cut uses local partial; missing TP peers staged at fixture setup. Single RTX3050."
    out=args.out/"canonical.json"
    def save():
        tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2));tmp.replace(out)
    save()
    torch.manual_seed(args.seed)
    raw=torch.nn.Module();h=4096;e=32
    for stage,n in (("w13",2*h),("w2",h)):
        for key,t in (("weight",torch.randint(0,256,(e,n,h//2),device="cuda",dtype=torch.uint8)),
                      ("weight_scale",torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)),
                      ("weight_scale_2",torch.linspace(.125,.5,e,device="cuda",dtype=torch.bfloat16))):
            setattr(raw,stage+"_"+key,torch.nn.Parameter(t,requires_grad=False))
    with cached_threshold_for_preparation(args.out):layer=ns["_prepare_humming_experts"](raw,torch)
    del raw,t
    norm=torch.linspace(.7,1.2,h,device="cuda",dtype=torch.bfloat16)
    boundary=scale_silu
    launches=[]
    if args.candidate=="route_ballot":
        folder=ROOT/"research_2026-11-18_rl_prefill"
        digest=hashlib.sha256((folder/"candidate.diff").read_bytes()).hexdigest()
        assert digest=="33a0a65719afafe691e5c685a7a32ead3194112fc18ce8434f3b31880ce9a721"
        report["candidate_sha256"]=digest
        proposed=original.module("paired_route_activation",folder/"route_activation.py")
        original_load=proposed.load_jit
        def relocate(*a,**kw):
            assert kw["cuda_files"]==["elementwise/route_activation.cuh"]
            kw["cuda_files"]=[str(folder/"route_activation.cuh")]
            return original_load(*a,**kw)
        proposed.load_jit=relocate
        native=proposed.route_activation_module()
        class CountNative:
            def run(self,*a):
                launches.append(a[0].shape[0])
                native.run(*a)
        proposed.route_activation_module=lambda: CountNative()
        boundary=proposed.scaled_prefill_activation
    report["unit"]="us"
    report["parity_tolerance"]="bitwise finalized BF16, finite outputs"
    small=adapt(split._original_split_expert_outputs,boundary);large=adapt(split._split_large_prefill_expert_outputs,boundary)
    fallback=adapt(ns["_humming_expert_outputs_default"],boundary)
    for requested in args.rows:
        # 16K full route slots require 3 GiB just for W13/gate activations. Use
        # 16 real 1024-row dispatches and include all final outputs on the 6GiB card.
        # True M16384 uses eager serial measurements when the 6GiB card fits
        # one full ranked workspace but cannot fit two simultaneous graph pools.
        torch.cuda.empty_cache()
        free=int(torch.cuda.mem_get_info()[0])
        full_estimate=requested*8*(3*4096)*2+requested*4096*2*5+512*2**20
        # Opt-in serial trial requires its FFN pooler/weight + a single 3-GiB
        # W13/gate scratch allocation. Static fixtures are reused in place.
        eager=requested>=16384 and (free>=full_estimate or args.serial_full)
        if args.serial_full:
            assert requested==16384 and args.envelope=="ffn"
            if free<requested*8*(3*4096)*2+768*2**20:
                report["deferred_16k_reason"]=f"free bytes {free} below one full Humming workspace plus FFN guard"
                save();return
        m=requested if requested<16384 or eager else 4096
        chunks=(requested+m-1)//m
        assert requested==m*chunks
        if args.envelope=="ffn":
            from dense_envelope import DenseEnvelope
            envelope=DenseEnvelope(m,args.seed)
        else: envelope=None
        input_width=8192 if envelope else h
        x=torch.empty((m,input_width),device="cuda",dtype=torch.bfloat16); ids=torch.empty((m,8),device="cuda",dtype=torch.int32)
        scores=torch.empty((m,8),device="cuda")
        def fixture(seed,pattern):
            gen=torch.Generator().manual_seed(seed)
            keys=torch.rand((requested,256),generator=gen)
            if pattern=="cluster":keys[::2,:8]+=2
            routes=keys.topk(8).indices
            ii=torch.where(routes<32,routes,-1).to(dtype=torch.int32,device="cuda")
            torch.manual_seed(seed)
            if args.serial_full:
                x.normal_(mean=0,std=.023)
                ids.copy_(ii)
                scores.uniform_();scores.div_(scores.sum(-1,keepdim=True))
                # Stage unrelated ranks without keeping an additional full test input or FP32 buffer.
                for offset in range(0,m,1024):
                    peer=torch.mm(x[offset:offset+1024],envelope.pool.t(),out_dtype=torch.float32)
                    envelope.assembled[offset:offset+1024].copy_(peer)
                return [(x,ids,scores,None)]
            xx=torch.randn((requested,input_width),device="cuda",dtype=torch.bfloat16)*.023
            ss=torch.rand((requested,8),device="cuda");ss/=ss.sum(-1,keepdim=True)
            result=[]
            for xx0,ii0,ss0 in zip(xx.split(m),ii.split(m),ss.split(m)):
                result.append((xx0,ii0,ss0,envelope.peer_fixture(xx0) if envelope else None))
            return result
        def load(batch):
            xx,ii,ss,peers=batch
            if args.serial_full: return
            x.copy_(xx);ids.copy_(ii);scores.copy_(ss)
            if envelope: envelope.stage_peers(peers)
        def execute(cand):
            inner=envelope.before(x) if envelope else x
            if cand and args.candidate=="compact_live":
                from compact_runtime import routed
                value=routed(layer,inner,ids,scores,norm,split,scale_silu)
                return envelope.after(value) if envelope else value
            if cand:
                fn=small if ids.numel()<=4096 or not getattr(layer,"_humming_large_prefill_partition_enabled",False) else large
                down,local=fn(fallback,layer,inner,ids,torch,256,defer_downscale=True)
            else:
                down,local=ns["_humming_expert_outputs"](layer,inner,ids,torch,global_experts=256,defer_downscale=True)
            value=finish(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=layer.w2_post_scale)
            return envelope.after(value) if envelope else value
        load(fixture(args.seed+1,"uniform")[0])
        graphs=[];outputs=[]
        for cand in (False,True):
            if eager:
                output=execute(cand)
                outputs.append(output);continue
            execute(cand);g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): output=execute(cand)
            graphs.append(g);outputs.append(output)
        memory_per_call=[]
        def timed(idx,fs):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            if len(fs)==1: load(fs[0])
            if args.serial_full:
                torch.cuda.reset_peak_memory_stats()
                before=int(torch.cuda.memory_allocated())
            a.record()
            for inp in fs:
                if len(fs)>1:load(inp)
                if eager: execute(bool(idx))
                else: graphs[idx].replay()
            b.record();b.synchronize()
            if args.serial_full:
                memory_per_call.append(dict(path="candidate" if idx else "original_or_control",
                    allocated_before=before,peak_allocated=int(torch.cuda.max_memory_allocated()),
                    peak_reserved=int(torch.cuda.max_memory_reserved())))
            return a.elapsed_time(b)*1000
        for pattern in args.patterns:
            name=f"M{requested}_{pattern}"
            record=dict(weight=1,times={k:[] for k in ("baseline","candidate","control")},seeds=[],order=[],parity=[],chunk_rows=m,chunks=chunks)
            record.update(execution="eager" if eager else "cuda_graph",free_at_shape_start_bytes=free,
                          tolerance="bitwise BF16",native_boundary_active=(8*m in launches) if args.candidate=="route_ballot" else True)
            if envelope:
                record["rank0_dense_projection_layout"]=envelope.kind
            report["bench"]["shapes"][name]=record
            for i in range(args.reps):
                seed=args.seed+17+31*i
                fs=fixture(seed,pattern)
                for inp in fs:
                    load(inp)
                    if eager:
                        outputs=[execute(False),execute(True)]
                    else:
                        for graph in graphs:graph.replay()
                    ok=torch.equal(outputs[0],outputs[1]) and bool(torch.isfinite(outputs[1]).all())
                    if not ok:
                        delta=(outputs[0].float()-outputs[1].float()).abs()
                        record["failed_parity"]=dict(maxabs=float(delta.max()),wrong=int((delta!=0).sum()))
                    record["parity"].append(ok);report["correct"]&=ok
                order=(0,1) if i%2==0 else (1,0)
                record["order"].append("AB" if i%2==0 else "BA");record["seeds"].append(seed)
                for v in order:
                    record["times"][("baseline","candidate")[v]].append(timed(v,fs))
                record["times"]["control"].append(timed(0,fs))
                save()
            t=record["times"];print(name,statistics.median(t["baseline"]),statistics.median(t["candidate"]),all(record["parity"]),flush=True)
            record["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        del output,outputs,graphs,fs
        if args.serial_full:
            report["serial_memory_per_measured_call"]=memory_per_call
            report["serial_note"]="Unchunked M16384, single eager arm at a time, real scales/FFN outputs and measured allocated/reserved high-water bytes; no CUDA graph speed or collectives implied."
        if not eager: del g,graph
        torch.cuda.empty_cache()
    if args.candidate=="route_ballot":
        report["active"]=bool(launches)
        report["native_boundary_capture_or_eager_rows"]=sorted(set(launches))
    report["complete"]=True;save()
if __name__=="__main__":main()

