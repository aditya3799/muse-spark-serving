"""Unchunked rank-local scored Humming FFN trial. No collective is simulated inside timing."""
from __future__ import annotations
import argparse
import ast
import ctypes as ct
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXPERTS=ROOT/"research_2026-11-18_rl_experts"
sys.path.insert(0,str(ROOT/"research_2026-11-18_lt_audit"))
BASE="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"

def setup_program():
    # Reuse the exact frozen activation constructor, layer packing, and native
    # Humming + traced group-8 call paths that qualified the serial 16K fixture.
    source=(EXPERTS/"bench.py").read_bytes()
    assert hashlib.sha256(source).hexdigest()=="07f9e7e94b89d7e2055c8b120aa731c94dbf447d59b52d216c7872e5904c1409"
    main=next(x for x in ast.parse(source).body if getattr(x,"name",None)=="main")
    def target(node,name):
        return isinstance(node,ast.For) and getattr(node.target,"id",None)==name
    begin=next(i for i,n in enumerate(main.body) if target(n,"p"))
    end=next(i for i,n in enumerate(main.body) if target(n,"requested"))
    body=main.body[begin:end]
    return compile(ast.Module(body=body,type_ignores=[]),str(EXPERTS/"bench.py"),"exec")

class LocalProjection:
    def __init__(self,torch,lib,weight,rows,prior,cap):
        from audit_match import describe
        self.torch,self.lib,self.weight,self.rows=torch,lib,weight,rows
        n,k=weight.shape
        self.shape=(rows,k)
        start=time.perf_counter()
        self.handle=lib.make_sweep(rows,n,n,k,n,cap[0],cap[1],4)
        assert self.handle
        matches=[(i,describe(lib,self.handle,i)) for i in range(lib.sweep_count(self.handle))]
        signature=("id","tile","split","reduction","swizzle","custom","stages","inner","cluster")
        good=[(i,a) for i,a in matches if all(a[k]==prior[k] for k in signature)]
        assert len(good)==1
        self.index,info=good[0]
        self.scratch=torch.empty(info["workspace"][1],device=weight.device,dtype=torch.uint8)
        self.calls=0
        self.discovery_s=time.perf_counter()-start

    def run(self,x):
        t=self.torch
        assert x.dtype==self.weight.dtype==t.bfloat16 and x.shape==self.shape
        assert x.device==self.weight.device and x.is_contiguous() and self.weight.is_contiguous()
        out=t.empty((self.rows,self.weight.shape[0]),dtype=t.float32,device=x.device)
        rc=self.lib.sweep_call(self.handle,self.index,1,self.weight.data_ptr(),x.data_ptr(),out.data_ptr(),
                                self.scratch.data_ptr() if self.scratch.numel() else 0,self.scratch.numel(),
                                t.cuda.current_stream().cuda_stream)
        assert rc==0
        self.calls+=1
        return out

    def measure_cold_screen(self, seed, cap):
        """Measure realistic setup cost; the wider trial keeps its independently frozen selection."""
        t=self.torch
        g=t.Generator(device=self.weight.device).manual_seed(seed)
        t.cuda.synchronize()
        start=time.perf_counter()
        temp=t.empty(cap,device=self.weight.device,dtype=t.uint8)
        examples=[]
        for mode in ("random","cancel","finite"):
            x=t.empty(self.shape,device=self.weight.device,dtype=t.bfloat16).normal_(generator=g)
            weight=t.empty_like(self.weight).normal_(std=.012,generator=g)
            if mode=="cancel":
                xx=x.view(self.rows,-1,2,256)
                ww=weight.view(weight.shape[0],-1,2,256)
                xx[:,:,1].copy_(xx[:,:,0]);ww[:,:,1].copy_(ww[:,:,0]).neg_()
                ww[::3,:,1,0].mul_(1.0078125)
            elif mode=="finite":
                x[:,::4]*=128
                weight[:,::4]/=128
                x[::11,1::7]=-0.0
            gold=t.mm(x,weight.t(),out_dtype=t.float32)
            examples.append((x,weight,gold))
        y=t.empty_like(gold)
        count=self.lib.sweep_count(self.handle)
        exact=0
        for i in range(count):
            ok=True
            for x,w,gold in examples:
                code=self.lib.sweep_call(self.handle,i,1,w.data_ptr(),x.data_ptr(),y.data_ptr(),
                                         temp.data_ptr(),cap,t.cuda.current_stream().cuda_stream)
                if code or not t.equal(y.view(t.int32),gold.view(t.int32)):
                    ok=False
                    break
            if not ok: continue
            exact+=1
            for _ in range(3):
                a,b=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
                a.record()
                code=self.lib.sweep_call(self.handle,i,1,w.data_ptr(),x.data_ptr(),y.data_ptr(),
                                         temp.data_ptr(),cap,t.cuda.current_stream().cuda_stream)
                y.to(t.bfloat16)
                b.record();b.synchronize();assert code==0
        t.cuda.synchronize()
        return dict(device_configs=count,exact=exact,search_seconds=self.discovery_s,
                    stress_screen_and_measure_seconds=time.perf_counter()-start)

    def close(self): self.lib.drop_sweep(self.handle)

class PackagedProjection:
    def __init__(self,torch,module,weight,rows):
        self.torch,self.module,self.weight=torch,module,weight
        t0=time.perf_counter()
        module.prepare(rows,weight)
        self.discovery_s=time.perf_counter()-t0
        key=(weight.device.index,rows,*weight.shape,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.are_deterministic_algorithms_enabled())
        self.selected=key in module._plans
        self.scratch=torch.empty(0,device=weight.device,dtype=torch.uint8)
        self.calls=0

    def run(self,x):
        self.calls+=int(self.selected)
        return self.module.qualified_target_row_mm(x,self.weight)

    def measure_cold_screen(self,seed,cap):
        return dict(actual_packaged_startup_seconds=self.discovery_s,qualified=self.selected)

    def close(self):
        pass

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--pairs",type=int,default=4)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--production",action="store_true",help="Use the packaged live runtime selector and its real registered custom op")
    args=p.parse_args()
    prog=setup_program()
    assert hashlib.sha256((ROOT/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest()==BASE
    assert args.pairs>=4 and args.pairs%2==0
    if args.preflight: return print("Scored setup, parent SHA and group-8 path verified")
    # Fresh directory per independent seed; setup writes its own provenance.
    args.out.mkdir(exist_ok=True,parents=True)
    setup_dir=args.out/"expert_fixture"
    setup_dir.mkdir(exist_ok=True)
    import importlib.util
    spec=importlib.util.spec_from_file_location("serial_full_experts",EXPERTS/"bench.py")
    expert=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(expert)
    init=vars(expert).copy()
    init["args"]=SimpleNamespace(out=setup_dir,seed=args.seed,candidate="route_ballot",envelope="ffn")
    exec(prog,init)
    torch=init["torch"]
    from torch.nn import functional as F
    from audit_match import load_lib
    from runtime_projection import native_workspace
    sys.path.insert(0,str(EXPERTS))
    from dense_envelope import DenseEnvelope
    m,localrows=16384,2048
    env=DenseEnvelope(m,args.seed)
    native=native_workspace(); assert native
    orig=load_lib(); lib=ct.CDLL(str(HERE/"sweep_fp32.so"))
    for name in ("make_sweep","sweep_count","sweep_attr","sweep_call","drop_sweep"):
        f,fn=getattr(orig,name),getattr(lib,name)
        fn.argtypes,fn.restype=f.argtypes,f.restype
    old=json.loads((HERE/"paired_stream/detail.json").read_text())
    assert old["device"]==torch.cuda.get_device_name()
    cap=(native[1],max(native[1],32*1024**2))
    if args.production:
        name="sglang.srt.layers.communication.muse_target_projection"
        spec=importlib.util.spec_from_file_location(name,HERE/"target_projection.py")
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        pool,unpool=[PackagedProjection(torch,module,getattr(env,k),localrows) for k in ("pool","unpool")]
    else:
        pool,unpool=[LocalProjection(torch,lib,getattr(env,k),localrows,old["shapes"][k+"_M16384_rows"]["selection"],cap) for k in ("pool","unpool")]
    cold={k:plan.measure_cold_screen(args.seed+47,cap[1]) for k,plan in (("pool",pool),("unpool",unpool))}
    torch.cuda.empty_cache()
    x=torch.empty(m,8192,device="cuda",dtype=torch.bfloat16)
    ids=torch.empty(m,8,device="cuda",dtype=torch.int32)
    scores=torch.empty(m,8,device="cuda")
    standard=init["ns"]["_humming_expert_outputs"]
    large,small,fallback=init["large"],init["small"],init["fallback"]
    layer,norm,finish,launches=init["layer"],init["norm"],init["finish"],init["launches"]
    def fixture(seed,pattern):
        gen=torch.Generator().manual_seed(seed)
        keys=torch.rand(m,256,generator=gen)
        if pattern=="cluster":keys[::2,:8]+=2
        pick=keys.topk(8).indices
        ids.copy_(torch.where(pick<32,pick,-1))
        g=torch.Generator(device="cuda").manual_seed(seed)
        x.normal_(mean=0,std=.023,generator=g)
        scores.uniform_(generator=g);scores.div_(scores.sum(-1,keepdim=True))
        # Seven inactive TP peers' outputs are setup data. This rank's own
        # contribution is computed in its original row position on every call.
        for start in range(localrows,m,1024):
            pc=torch.mm(x[start:start+1024],env.pool.t(),out_dtype=torch.float32)
            env.assembled[start:start+1024].copy_(pc)

    def execute(arm,verify=False):
        vendor=arm in ("dense","joint")
        if vendor:
            raw=pool.run(x[:localrows])
            if verify:
                expected=torch.mm(x[:localrows],env.pool.t(),out_dtype=torch.float32)
                assert torch.equal(raw.view(torch.int32),expected.view(torch.int32))
                del expected
            own=raw.to(torch.bfloat16)
            env.assembled[:localrows].copy_(own)
            del raw,own
            inner=F.rms_norm(env.assembled,(4096,),env.norm_weight,1e-5).to(torch.bfloat16)
        else: inner=env.before(x)
        if arm=="joint":
            fn=large if getattr(layer,"_humming_large_prefill_partition_enabled",False) else small
            down,loc=fn(fallback,layer,inner,ids,torch,256,defer_downscale=True)
        else:
            down,loc=standard(layer,inner,ids,torch,global_experts=256,defer_downscale=True)
        value=finish(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,expert_output_scale=layer.w2_post_scale)
        if not vendor:return env.after(value)
        u=unpool.run(value[:localrows])
        if verify:
            expected=torch.mm(value[:localrows],env.unpool.t(),out_dtype=torch.float32)
            assert torch.equal(u.view(torch.int32),expected.view(torch.int32))
        return u.to(torch.bfloat16)

    detail=dict(device=torch.cuda.get_device_name(),seed=args.seed,rows=m,local_rows=localrows,scope="Full rank-local FFN; other seven ranks' pooled rows staged before timing. EP sum is local partial. No network, router, attention or residual. Eager one arm at a time.",
                cold_tuning=cold,discovered_seconds=dict(pool=pool.discovery_s,unpool=unpool.discovery_s),
                workspace_bytes=dict(pool=pool.scratch.numel(),unpool=unpool.scratch.numel()),
                arms=["baseline","dense","joint","control"],shapes={},complete=False)
    paths={"dense":"full_ffn_vendor","joint":"full_ffn_vendor_group8"}
    if args.production: paths={k:v+"_packaged" for k,v in paths.items()}
    def save():
        (args.out/"detail.json").write_text(json.dumps(detail,indent=2)+"\n")
        for key,tag in paths.items():
            report=dict(agent="prefill",candidate=tag,scope=detail["scope"],baseline_sha256=BASE,seed=args.seed,
                        correct=True,active=(pool.calls>0 or unpool.calls>0) if key=="dense" else bool(launches),
                        complete=detail["complete"],bench={"instances":1,"shapes":{}})
            for name,r in detail["shapes"].items():
                report["correct"] &= all(r["parity"][key]) and all(r["parity"]["control"])
                report["bench"]["shapes"][name]=dict(weight=1,times={label:r["times"][arm] for label,arm in (("baseline","baseline"),("candidate",key),("control","control"))})
            (args.out/(key+".json")).write_text(json.dumps(report,indent=2)+"\n")
    fixture(args.seed+1,"uniform")
    for arm in ("baseline","dense","joint"):execute(arm,verify=True)
    if args.production:
        for arm in ("control","joint","dense","baseline","baseline","dense","joint","control"):execute(arm)
        detail["production_packaged"]=True
        detail["qualified_legs"]={"pool":pool.selected,"unpool":unpool.selected}
    torch.cuda.synchronize()
    assert launches
    previous=None
    for pattern in ("uniform","cluster"):
        name="M16384_"+pattern
        r=dict(times={key:[] for key in detail["arms"]},order=[],parity={k:[] for k in ("dense","joint","control")},input_changed=[],memory_peak_allocated=0,fixture_setup_us=[])
        detail["shapes"][name]=r
        for i in range(args.pairs):
            t0,t1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            t0.record();fixture(args.seed+31*i+11,pattern);t1.record();t1.synchronize()
            r["fixture_setup_us"].append(t0.elapsed_time(t1)*1000)
            # Verify raw FP32 on changed data outside the timed FFN, and use
            # this rank's real normalized Humming output as unpooler input.
            ref=execute("baseline")
            for arm in ("dense","joint","control"):
                y=execute(arm,verify=True)
                r["parity"][arm].append(bool(torch.equal(y.view(torch.int16),ref.view(torch.int16))) and bool(torch.isfinite(y).all()))
                del y
            assert all(v[-1] for v in r["parity"].values())
            finger=ref[:4,:128].clone()
            if previous is not None: assert not torch.equal(previous,finger)
            r["input_changed"].append(previous is None or not torch.equal(previous,finger))
            previous=finger
            del ref
            order=["baseline","dense","joint","control"] if i%2==0 else ["control","joint","dense","baseline"]
            r["order"].append(order)
            for arm in order:
                torch.cuda.reset_peak_memory_stats()
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();execute(arm);b.record();b.synchronize()
                r["times"][arm].append(a.elapsed_time(b)*1000)
                r["memory_peak_allocated"]=max(r["memory_peak_allocated"],int(torch.cuda.max_memory_allocated()))
            save()
            print(pattern,i,{k:round(v[-1]/1000,2) for k,v in r["times"].items()},flush=True)
    detail.update(complete=True,active_group8_launches=len(launches),active_vendor_calls=[pool.calls,unpool.calls])
    save()
    pool.close();unpool.close()

if __name__=="__main__":main()

