"""Exercise the patched lifecycle with real CPU tensors and a mutable native-planner mock."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as S, MethodType

import torch

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
parent = H.parent/"decode_qkv/parent_flashattention_backend.py"
src = H/"flashattention_backend.py"
def functions(p):
    node = next(n for n in ast.parse(p.read_text()).body if getattr(n,"name",None)=="FlashAttentionBackend")
    return {n.name:n for n in node.body if isinstance(n,ast.FunctionDef)}
old,new=functions(parent),functions(src)
def compiled(nodes):
    return compile(ast.fix_missing_locations(ast.Module(ast.parse("from __future__ import annotations").body+nodes,[])),str(src),"exec")
names=["_prepare_swa_verify_scheduler","_swa_verify_scheduler_for_layer","init_forward_metadata_in_graph"]
gl={"torch":torch}
exec(compiled([new[n] for n in names]),gl)
def no_additions():
    class Remove(ast.NodeTransformer):
        def visit_If(self,node):
            t=ast.unparse(node.test)
            if t in ("self._swa_verify_enabled","self._swa_verify_enabled and forward_batch.forward_mode.is_target_verify()","is_swa_layer and metadata.swa_verify_scheduler is not None"):
                return None
            return self.generic_visit(node)
    for name in ("init_forward_metadata","init_forward_metadata_in_graph","forward_extend"):
        assert ast.dump(Remove().visit(copy.deepcopy(new[name])))==ast.dump(old[name]),name
    for n in old.keys() - {"__init__","init_forward_metadata","init_forward_metadata_in_graph","forward_extend"}:
        assert ast.dump(old[n])==ast.dump(new[n]),n
no_additions()

class NativeMock:
    def __init__(self):
        self.tape=None
        self.calls=[]
    def __call__(self,**kw):
        self.calls.append(kw)
        n=kw["batch_size"]
        r=(n+3)//4*4
        assert kw["cu_seqlens_k_new"] is None and kw["max_seqlen_k_new"]==0
        assert kw["pack_gqa"] is None and kw["sm_margin"]==0 and kw["causal"]
        assert kw["window_size"][0]<kw["max_seqlen_k"]-1 and kw["window_size"][1]==0
        assert kw["cache_seqlens"].shape==(n,)
        # H200 prepared local layout: three aligned regions and mutable semaphore.
        result=torch.empty(3*r+1,dtype=torch.int32)
        def prepare():
            result.zero_()
            result[:n]=kw["cache_seqlens"]
            result[r:r+n]=torch.diff(kw["cu_seqlens_q"])
            result[2*r:2*r+n]=kw["num_splits"]
        prepare()
        if self.tape is not None:self.tape.append(prepare)
        return result

def fixture(b=4,width=8,cap=65536,splits=0,dtype=torch.float8_e4m3fn):
    fn=NativeMock()
    layer=S(is_cross_attention=False,tp_q_head_num=16,tp_k_head_num=2,head_dim=64,v_head_dim=64,logit_cap=0)
    f=S(batch_size=b,spec_info=S(ragged_verify_layout=None,custom_mask=None,num_tokens_per_req=width),
        encoder_lens=None,positions=torch.arange(b*width),forward_mode=S(is_target_verify=lambda:True,is_draft_extend_v2=lambda:False))
    m=S(swa_verify_scheduler=None,image_token_ranges=None,swa_spec_metadata=None,max_seq_len_q=width,
        cache_seqlens_int32=torch.arange(b,dtype=torch.int32)*100+4096,cu_seqlens_q=torch.arange(b+1,dtype=torch.int32)*width,
        cu_seqlens_k=torch.arange(b+1,dtype=torch.int32)*4096,
        page_table=torch.empty((b,cap),dtype=torch.int32),swa_page_table=torch.empty((b,cap),dtype=torch.int32))
    backend=S(_swa_verify_enabled=True,forward_metadata=m,use_sliding_window_kv_pool=True,
              speculative_num_draft_tokens=width,page_size=1,num_attention_heads=16,num_kv_heads=2,head_dim=64,
              kv_cache_dtype=dtype,has_softcap=False,num_splits=splits,sliding_window_size=2047,
              _capture_unified_verify_metadata=lambda f:False,_get_scheduler_metadata=fn)
    for name in names:setattr(backend,name,MethodType(gl[name],backend))
    kwargs=dict(metadata=m,forward_batch=f,layer=layer,q=torch.empty(b*width,1024,dtype=dtype),
                key_cache=torch.empty(3,1,2,64,dtype=dtype),value_cache=torch.empty(3,1,2,64,dtype=dtype),
                page_table=m.swa_page_table,cache_seqlens=m.cache_seqlens_int32,cu_seqlens_q=m.cu_seqlens_q,
                cu_seqlens_k=m.cu_seqlens_k,max_seqlen_q=width,window_size=(2047,0),causal=True,cp_active=False,
                use_local_attn=False,use_cascade_attn=False,sinks=None,score_mod=None,rel_bias=None)
    return fn,backend,m,f,kwargs

# Replay previously recorded native operation on changed data. Each simulated
# attention call reads the selected plan and applies the pinned reset contract.
captures=replays=consumers=0
for b,width,cap,splits in [(1,8,8192,0),(4,8,32768,1),(13,7,65536,4),(32,8,131072,0)]:
    fn,backend,m,f,kw=fixture(b,width,cap,splits)
    tape_records=[]
    for iteration in range(3):
        fn.tape=[]
        backend.init_forward_metadata_in_graph(f)
        result=backend._swa_verify_scheduler_for_layer(**kw)
        assert result is not None and result.numel()==3*((b+3)//4*4)+1
        assert fn.calls[-1]["cache_seqlens"] is m.cache_seqlens_int32
        assert fn.calls[-1]["cu_seqlens_q"] is m.cu_seqlens_q
        assert fn.calls[-1]["max_seqlen_k"]==cap and fn.calls[-1]["qkv_dtype"]==torch.float8_e4m3fn
        assert fn.calls[-1]["num_splits"]==splits
        for tag in range(46):
            assert result is backend._swa_verify_scheduler_for_layer(**kw)
        assert len(fn.tape)==1
        tape_records.append((fn.tape[0],result))
        captures+=1
    assert len({v.data_ptr() for _,v in tape_records})==3 and len(fn.calls)==3
    for step in range(9):
        m.cache_seqlens_int32.copy_(torch.arange(b)*113+(b+step)*137)
        for prepare,out in tape_records:
            out.fill_(-777);ptr=out.data_ptr()
            prepare()
            assert torch.equal(out[:b],m.cache_seqlens_int32) and out.data_ptr()==ptr
            for layer_id in range(46):
                assert out[-1].item()==0
                out[-1]+=23
                # Native combine reset for split, explicit last-element zero for no split.
                out[-1].zero_()
                consumers+=1
            replays+=1
    assert len(fn.calls)==3
fn,b,m,f,kw=fixture()
checks=[]
for obj,key,val in [
    (b,"_swa_verify_enabled",False),(f.spec_info,"custom_mask",object()),(f.spec_info,"ragged_verify_layout",object()),
    (f.spec_info,"num_tokens_per_req",4),(b,"speculative_num_draft_tokens",7),(f,"positions",torch.arange(31)),
    (f,"encoder_lens",object()),(m,"image_token_ranges",object()),(m,"swa_spec_metadata",object()),
    (m,"swa_page_table",None),(m,"swa_page_table",torch.empty(4,2048)),(m,"swa_page_table",torch.empty(3,65536)),
    (m,"cu_seqlens_q",torch.arange(3)),(m,"cache_seqlens_int32",torch.arange(4,dtype=torch.int64)),
    (m,"cu_seqlens_q",torch.arange(10,dtype=torch.int32)[::2]),(f.forward_mode,"is_target_verify",lambda:False)]:
    save=getattr(obj,key);setattr(obj,key,val)
    b._prepare_swa_verify_scheduler(m,f);assert m.swa_verify_scheduler is None,key
    setattr(obj,key,save);checks.append(key)
assert not fn.calls
b._prepare_swa_verify_scheduler(m,f)
for name,value in [
    ("q",torch.empty(31,1024,dtype=b.kv_cache_dtype)),("q",torch.empty(32,512,dtype=b.kv_cache_dtype)),
    ("page_table",m.swa_page_table.clone()),("cache_seqlens",m.cache_seqlens_int32.clone()),
    ("cu_seqlens_q",m.cu_seqlens_q.clone()),("cu_seqlens_k",m.cu_seqlens_k.clone()),
    ("max_seqlen_q",5),("window_size",(2047,2047)),("window_size",(-1,-1)),("causal",False),
    ("use_cascade_attn",True),("use_local_attn",True),("cp_active",True),("sinks",object()),
    ("score_mod",object()),("rel_bias",object()),("key_cache",kw["key_cache"].to(torch.bfloat16)),
    ("key_cache",torch.empty(3,2,1,64,dtype=b.kv_cache_dtype)),("value_cache",kw["value_cache"][1:])]:
    modified=dict(kw);modified[name]=value
    assert b._swa_verify_scheduler_for_layer(**modified) is None,name
    checks.append(name)
for obj,key,val in [(b,"num_splits",7),(kw["layer"],"tp_q_head_num",8),(kw["layer"],"tp_k_head_num",1),
                    (kw["layer"],"v_head_dim",128),(kw["layer"],"head_dim",128),(kw["layer"],"logit_cap",3),
                    (kw["layer"],"is_cross_attention",True)]:
    save=getattr(obj,key);setattr(obj,key,val)
    assert b._swa_verify_scheduler_for_layer(**kw) is None,key
    setattr(obj,key,save);checks.append(key)
assert len(fn.calls)==1

# Dispatch through the exact scored forward_extend conditional and its FA3 call.
oldcall=[n for n in ast.walk(old["forward_extend"]) if isinstance(n,ast.Call) and any(x.arg=="scheduler_metadata" for x in n.keywords)]
newcall=[n for n in ast.walk(new["forward_extend"]) if isinstance(n,ast.Call) and any(x.arg=="scheduler_metadata" for x in n.keywords)]
assert len(oldcall)==len(newcall)==1 and ast.dump(oldcall[0])==ast.dump(newcall[0])
dispatch=next(n for n in ast.walk(new["forward_extend"]) if isinstance(n,ast.If) and "metadata.scheduler_metadata is not None" in ast.unparse(n.test))
m.scheduler_metadata=object();b.topk=1
for is_swa,custom,expected in [(False,None,m.scheduler_metadata),(True,None,m.swa_verify_scheduler[0]),(True,object(),None)]:
    ns={**kw,"self":b,"sched_meta":None,"is_swa_layer":is_swa,"score_mod":custom}
    exec(compiled([dispatch]),ns)
    assert ns["sched_meta"] is expected
# Verify the native binary's public source and local archived component receipts.
archive=ROOT/"swarm_2026-11-16/attention_speculation/phase2/swa_scheduler_reuse"
contract=json.loads((archive/"NATIVE_CONTRACT_REVIEW.json").read_text())
cpp=archive/"native_run/source_snapshots/08_flash_api.cpp"
assert hashlib.sha256(cpp.read_bytes()).hexdigest()==contract["native_sources"]["flash_api.cpp"]["sha256"]
native=cpp.read_text()
assert "run_mha_fwd_combine(params, stream" in native and "params.tile_count_semaphore_offset + 1)}).zero_()" in native
record=json.loads((archive/"native_run/results.json").read_text())
assert record["passed"] and all(row["passed"] for row in record["rows"])
proof=dict(patch=json.loads((H/"build.json").read_text()),captures=captures,changed_input_replays=replays,
           sequential_consumer_checks=consumers,negative_guard_fixtures=len(checks),actual_parent_fa_arguments_unchanged=True,
           patched_dispatch_checks=3,parent_lifecycle_ast_restored_after_removing_additions=True,
           archived_native_device=record["device"],archived_native_checks=sum(row["checks"] for row in record["rows"]),
           archived_component_speedups=[row["speedup"] for row in record["rows"]],
           scope="Current patch has source and CPU-recorded lifecycle validation. The archived native GPU test covers the same FA3 reuse primitive through a previous integration using BF16. No native FP8 serving test.")
(H/"verification.json").write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
