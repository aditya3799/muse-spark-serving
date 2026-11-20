"""Exercise patched runner/model/TP methods and production Torch custom op on live CUDA streams."""
from __future__ import annotations
import argparse
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS,MethodType

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]; PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
prefix="python/sglang/"
paths=[prefix+"srt/model_executor/runner/prefill_cuda_graph_runner.py",prefix+"srt/models/muse_spark.py",
       prefix+"srt/layers/communication/muse_spark_gemm_ag.py",prefix+"srt/layers/communication/muse_target_projection.py",
       prefix+"srt/layers/communication/muse_target_projection.cc"]
def method(path,name,binding):
    nodes=[n for n in ast.walk(ast.parse(path.read_bytes())) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1
    mod=ast.parse("from __future__ import annotations").body+nodes
    exec(compile(ast.Module(body=mod,type_ignores=[]),str(path),"exec"),binding)
    return binding[name]
def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT/"optimization_marlin_tiles_2026-11-15"))
    from kernel_runtime import bootstrap
    bootstrap(ROOT/"sglang")
    import torch
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp)
        for rel in paths:
            p=base/rel;p.parent.mkdir(exist_ok=True,parents=True)
            op=subprocess.run(["git","-C",str(ROOT/"sglang"),"show",PIN+":"+rel],capture_output=True)
            if op.returncode==0:p.write_bytes(op.stdout)
        subprocess.run(["git","-C",temp,"apply",*["--include="+p for p in paths],str(HERE/"candidate.diff")],check=True)
        fullname="sglang.srt.layers.communication.muse_target_projection"
        spec=importlib.util.spec_from_file_location(fullname,base/paths[3])
        proj=importlib.util.module_from_spec(spec);sys.modules[fullname]=proj;spec.loader.exec_module(proj)
        torch.manual_seed(226491)
        m,local=16384,2048
        pool=torch.nn.Linear(8192,4096,bias=False,device="cuda",dtype=torch.bfloat16).requires_grad_(False)
        unpool=torch.nn.Linear(4096,8192,bias=False,device="cuda",dtype=torch.bfloat16).requires_grad_(False)
        q=NS(is_moe_layer=True,mlp=NS(experts=NS(use_humming=True),pre_expert_proj=pool,post_expert_proj=unpool))
        mockmodel=NS(get_layers=lambda:[q,q])
        parallel=NS(tp_size=8,attn_dp_size=1,attn_cp_size=1)
        fake_env=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:False))
        orig=method(base/paths[1],"prepare_target_projection_plans",dict(torch=torch,get_parallel=lambda:parallel,envs=fake_env))
        orig(mockmodel,m)
        assert not proj._seen and torch.cuda.get_device_capability()==(8,6)
        # Exact model gate above rejects SM86. Force only its capability predicate for integration testing.
        proxy=NS(inference_mode=torch.inference_mode,cuda=NS(is_available=lambda:True,get_device_capability=lambda:(9,0)))
        warm=method(base/paths[1],"prepare_target_projection_plans",dict(torch=proxy,get_parallel=lambda:parallel,envs=fake_env))
        events=[]
        def preparation(rows):
            events.append("model projection warmup")
            warm(mockmodel,rows)
        mockmodel.prepare_target_projection_plans=preparation
        gr=NS(world_size=8,rank_in_group=0,all_gather=lambda x,dim:x)
        mm=method(base/paths[2],"prefill_row_projection",dict(torch=torch))
        x=torch.empty(m,8192,device="cuda",dtype=torch.bfloat16).normal_()
        z=torch.empty(m,4096,device="cuda",dtype=torch.bfloat16).normal_()
        def forward(batch,n):
            assert n==m
            return (mm(gr,x,pool.weight,qualify_vendor=True),mm(gr,z,unpool.weight,qualify_vendor=True))
        stream=torch.cuda.Stream()
        class Backend:
            def capture_one(self,tag,run_once,**kwargs):
                events.append("backend capture")
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run_once()
                torch.cuda.current_stream().wait_stream(stream)
                self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph,stream=stream):self.out=run_once()
        back=Backend()
        runner=NS(model_runner=NS(model=mockmodel),backend=back,enable_cp_bcg_capture=False,
                  capture_prepare=lambda n:(NS(lora_ids=None,global_num_tokens_gpu=None),NS()),
                  _is_full_backend=False,_capture_lora=False,_init_forward_metadata_for_capture=lambda *a:None,
                  _run_forward=forward)
        runner_method=method(base/paths[0],"capture_one_shape",dict(ShapeKey=lambda **kw:NS(**kw),nullcontext=nullcontext,BreakableCudaGraphBackend=type("unused",(),{})))
        with torch.inference_mode():
            runner_method(runner,m)
            assert events==["model projection warmup","backend capture"]
            assert len(proj._seen)==2
            checks=[]
            prev=None
            original_calls=0
            for i in range(5):
                x[:local].normal_(std=(i+1)/10)
                z[:local].normal_(std=(i+1)/10)
                if i%2:
                    x[:local,1::13]=-0.0
                    z[:local,1::7]=0
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):back.graph.replay()
                torch.cuda.current_stream().wait_stream(stream)
                truths=[mm(gr,v,w,qualify_vendor=False) for v,w in [(x,pool.weight),(z,unpool.weight)]]
                row=[bool(torch.equal(a.view(torch.int16),b.view(torch.int16))) for a,b in zip(back.out,truths)]
                checks.append(row)
                val=back.out[0][:1].clone()
                if prev is not None:assert not torch.equal(prev,val)
                prev=val
                assert all(row),row
            # An unseen contiguous shape, alternative dtype, and offset pointer must fall back to native.
            fallback=[]
            for v,w in ((x[:39],pool.weight),(x[:19].half(),pool.weight.half()),
                        (x.view(-1)[1:1+19*x.shape[1]].view(19,-1),pool.weight)):
                y=proj.qualified_target_row_mm(v,w);ref=torch.mm(v,w.t(),out_dtype=torch.float32)
                fallback.append(torch.equal(y.view(torch.int32),ref.view(torch.int32)))
            assert all(fallback)
            report=dict(device=torch.cuda.get_device_name(),real_capability=torch.cuda.get_device_capability(),
                        mock_h200_model_gate=True,actual_arch_gate_rejected=True,runner_events=events,
                        qualified_keys=[list(k) for k in proj._plans],seen_keys=[list(k) for k in proj._seen],
                        capture_replay_checks=checks,unknown_dtype_shape_alignment_fallback=fallback,
                        transport="Rank-zero row projection, identity gather for unavailable peers",production_selector_active=bool(proj._plans))
            args.out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report),flush=True)
            assert report["production_selector_active"],"Selector kept both native; no production fast path was measured"
if __name__=="__main__":main()

