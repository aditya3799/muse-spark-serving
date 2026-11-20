"""Execute the production route builder, W2 scatter, serving wrapper and finalizer on SM86.

Inject a known accumulator pattern in place of Hopper MMA. The independent reference
constructs per-token results directly from local IDs and never uses the route builder.
"""
import argparse
import ast
import ctypes as ct
import importlib
import json
from pathlib import Path
import subprocess
import sys
import torch
import triton
import triton.language as tl
from bench_experts import ROOT,SOURCE,module
from build_hopper import FLAGS,materialize


@triton.jit
def _reference(X,IDS,Y,B:tl.constexpr):
    slot=tl.program_id(0)
    col=tl.program_id(1)*B+tl.arange(0,B)
    expert=tl.load(IDS+slot)
    if expert>=0:
        h=tl.load(X+(slot//8)*4096+col).to(tl.float32)
        val=h*((expert%4)+1)+(expert+1)*0.03125+(slot%8)*0.125
        tl.store(Y+slot*4096+col,val)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    out=p.parse_args().out.resolve()
    source=Path(__file__).with_name("native_package_02").joinpath("hopper_expert.cu").read_text()
    cut=source.index("constexpr int UNIT_BYTES")
    code=source[:cut]+"__device__ void early_issue(const Params&, Smem&) {}\n"
    k0=source.index("__device__ __forceinline__ int wgs_kperm3_col")
    code+=source[k0:source.index("// phase A's x gather",k0)]
    ep0=source.index("template <bool PB>\n__device__ __forceinline__ void wgs_epi_tile")
    code+=source[ep0:source.index("template <bool PB>\n__device__ __forceinline__ void wgs_flush",ep0)]
    code+=Path(__file__).with_name("route_fixture.cuh").read_text()
    cu=out/"route.cu"
    cu.write_text(code)
    so=out/"route.so"
    subprocess.run(["nvcc","-std=c++17","-O3","-arch=sm_86","--shared","-Xcompiler=-fPIC","--cudart=shared",
                    *FLAGS,str(cu),"-o",str(so)],check=True)
    sys.path.insert(0,str(Path(__file__).parent))
    backend=importlib.import_module("native_package_02.hopper_backend")
    sys.path.insert(0,str(ROOT/"optimization_marlin_tiles_2026-11-15"))
    from kernel_runtime import bootstrap
    bootstrap(ROOT/"sglang")
    moe=SOURCE/"kernels/ops/moe"
    module("sglang.kernels.ops.moe.muse_spark_humming_scale",moe/"muse_spark_humming_scale.py")
    finish=module("scored_finalize",moe/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
    helper=ast.parse(Path(__file__).with_name("native_package_02").joinpath("muse_spark_nvfp4_moe.py").read_text())
    fn=next(n for n in helper.body if getattr(n,"name",None)=="_humming_expert_outputs")
    import types
    srtname="sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe"
    obj=types.ModuleType(srtname)
    obj.__package__="native_package_02"
    exec(compile(ast.Module([fn],[]),"packed adapter","exec"),obj.__dict__)
    sys.modules[srtname]=obj
    methods=ast.parse((SOURCE/"srt/layers/muse_spark/muse_spark_fp4_moe.py").read_text())
    cls=next(n for n in methods.body if isinstance(n,ast.ClassDef) and any(getattr(f,"name",None)=="_marlin_expert_outputs" for f in n.body))
    keep={"forward","_marlin_expert_outputs","_forward_marlin"}
    cls.body=[n for n in cls.body if getattr(n,"name",None) in keep]
    cls.bases=[]
    cls.decorator_list=[]
    import __future__
    ns={"torch":torch,"fused_post_expert_norm_weighted_sum":finish}
    exec(compile(ast.Module([cls],[]),"scored forward","exec",flags=__future__.annotations.compiler_flag),ns)
    obj=ns[cls.name]()
    obj.use_marlin=obj.use_humming=True
    obj.global_num_experts=256
    obj._expert_map=torch.full((256,),-1,device="cuda",dtype=torch.int32)
    obj._expert_map[64:96]=torch.arange(32,device="cuda")
    obj._expert_lookup=torch.cat((obj._expert_map[:1],obj._expert_map))
    obj._post_expert_norm=object()
    obj._post_expert_norm_weight=torch.linspace(.71,1.24,4096,device="cuda").bfloat16()
    obj._post_expert_norm_eps=1e-8
    obj._post_expert_norm_weight_first=False
    l=backend.HopperExperts()
    l.lib=ct.CDLL(str(so))
    l.lib.expert_run.argtypes=[ct.c_void_p]*10+[ct.c_int]*3+[ct.c_void_p]
    l.w13=l.w2=l.table=l.gate_scale=torch.zeros(1,device="cuda")
    l.w2_post_scale=torch.linspace(.25,1.25,32,device="cuda").bfloat16()
    l.experts=32
    l.grid=16
    l.sizes=(608,1408,30932992)
    obj._humming_layer=l
    torch.manual_seed(6751)
    results=[]
    for m,pat in ((0,"empty"),(1,"remote"),(8,"hot"),(128,"spread"),(384,"hot"),(385,"boundary"),(753,"spread"),(16384,"spread")):
        ids=torch.randint(0,256,(m,8),device="cuda",dtype=torch.int32)
        if pat=="remote": ids.fill_(12)
        if pat=="hot": ids[:,::2]=64
        if pat=="boundary": ids[-1]=95
        if m: ids[::31,0]=-1
        x=torch.randn((m,4096),device="cuda",dtype=torch.bfloat16)
        scores=torch.rand((m,8),device="cuda")
        def forward():
            return obj.forward(x,scores,ids,obj._post_expert_norm)
        def reference():
            local=obj._expert_lookup[ids.clamp_min(-1)+1]
            down=x.new_empty((m,8,4096))
            if m: _reference[(m*8,16)](x,local,down,256)
            return finish(down,scores,norm_weight=obj._post_expert_norm_weight,eps=1e-8,
                          weight_first=False,slot_expert_ids=local,expert_output_scale=l.w2_post_scale)
        actual=forward()
        assert torch.equal(actual,reference()),(m,pat)
        if m==128:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                buf=forward()
            for i in range(3):
                ids.random_(0,256)
                x.normal_()
                graph.replay()
                assert torch.equal(buf,reference()),i
        results.append(dict(rows=m,pattern=pat,finalized_elements=actual.numel()))
        (out/"results.json").write_text(json.dumps(dict(rows=results,device=torch.cuda.get_device_name(),
            scope="Real native routing and raw W2 scatter; synthetic projection accumulators; scored forward/finalizer."),indent=2))

if __name__=="__main__":
    main()

