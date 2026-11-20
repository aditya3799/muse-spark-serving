from pathlib import Path
import json
import ast
import sys
import types
from types import SimpleNamespace as S
from contextlib import contextmanager
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parent
def funcs(path,names,scope):
    tree=ast.parse((HERE/path).read_bytes())
    ns=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(ns)==len(names)
    exec(compile(ast.Module(ast.parse("from __future__ import annotations").body+ns,[]),str(path),"exec"),scope)
    return scope

def main():
    flags=S(sp=S(enabled=False))
    ctx=S(enable_layernorm_sp=True,attn_cp_size=1,moe_dp_size=1,tp_group=S(world_size=8))
    model=S(hf_config=S(architectures=["MuseSparkForConditionalGeneration"]),hf_text_config=S(hidden_size=8192,moe_hidden_size=4096))
    execbag=S(moe=S(moe_runner_backend="humming"))
    det=S(get=lambda:False)
    mode=S(EXTEND=1)
    modules={"sglang.srt.runtime_context":S(get_exec=lambda:execbag),"sglang.srt.environ":S(envs=S(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=det)),
             "sglang.srt.model_executor.forward_batch_info":S(ForwardMode=mode)}
    torch_mock=S(cuda=S(is_available=lambda:True,get_device_capability=lambda:(9,0)))
    ns=dict(get_flags=lambda:flags,get_parallel=lambda:ctx,torch=torch_mock,MUSE_SP_ARCHITECTURES={"MuseSparkForConditionalGeneration"},
            SP_SUPPORTED_ARCHITECTURES={"MuseSparkForConditionalGeneration","Qwen3ForCausalLM"})
    funcs("layernorm_sp.py",("initialize_layernorm_sp","is_muse_spark_architecture","layernorm_sp_enabled","runs_sp"),ns)
    cases={}
    with patch.dict(sys.modules,modules):
        ns["initialize_layernorm_sp"](model_config=model)
        for mode_id,rows,expected in ((1,512,True),(1,520,True),(1,16384,True),(1,32,False),(1,753,False),(2,16384,False),(1,None,False)):
            assert ns["runs_sp"](mode_id,rows)==expected;cases[f"mode{mode_id}_m{rows}"]=expected
        for raw,returned,expected in ((753,96,True),(753,753,False),(511,64,True),(511,511,False),(16384,2048,True)):
            assert ns["runs_sp"](1,raw,returned_rows=returned)==expected
            cases[f"body_returned_{returned}_for_raw_{raw}"]=expected
        for key,obj,attr,val in (("cp",ctx,"attn_cp_size",2),("expert_dp",ctx,"moe_dp_size",2),("deterministic",det,"get",lambda:True),
              ("non_hopper",torch_mock.cuda,"get_device_capability",lambda:(8,6)),("bad_width",model.hf_text_config,"moe_hidden_size",32)):
            saved=getattr(obj,attr);setattr(obj,attr,val);ns["initialize_layernorm_sp"](model_config=model)
            assert not flags.sp.enabled;cases[key]=False;setattr(obj,attr,saved)
    m=16384;h=8192;e=4096;tp=8
    def send(width,bytes_=2):return m*width*bytes_*(tp-1)/tp/2**20
    calc={"unit":"binary MiB sent per rank in logical ring model; no NCCL timing",
          "parent":{"o_ar":2*send(h),"pool_ag":send(e),"experts_ar":2*send(e),"unpool_ag":send(h)},
          "gathered_router":{"qkv_ag":send(h),"o_rs":send(h),"router_input_ag":send(h),"pool_ag":send(e),"experts_rs":send(e)},
          "sharded_router":{"qkv_ag":send(h),"o_rs":send(h),"route_weights+ids_ag":send(8,8),"pool_ag":send(e),"experts_rs":send(e)},
          "once_per_forward":{"five_taps_bf16_ag":send(5*h),"last_hidden_bf16_ag":send(h)}}
    for name in ("parent","gathered_router","sharded_router","once_per_forward"):
        calc[name]["total"]=sum(calc[name].values())
    (HERE/"dispatch_and_traffic.json").write_text(json.dumps(dict(cases=cases,traffic=calc),indent=2))
    print(cases,calc)
if __name__=="__main__":main()

