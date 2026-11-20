"""Check patched SGLang graph admission/marker wiring and two-step retained buffer lifetime."""
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
import torch

H=Path(__file__).resolve().parent;R=H.parent;K=R/"research_2026-11-18_rl/lattice_tree"
sys.path.insert(0,str(K));from fixed_tree_adapter import TreeStep
assert os.environ.get("TRITON_INTERPRET")=="1" and not torch.cuda.is_available()
par=N(tp_group=N(world_size=1),attn_dp_enabled=False,attn_dcp_size=1,enable_prefill_cp=False)
scope=dict(SIMULATE_ACC_LEN=0,get_parallel=lambda:par,get_dp_tp_group=lambda:par.tp_group,IdSpaceKind=N(SLIDING_WINDOW=2),torch=torch,TreeStep=TreeStep,
           WIDTH=8,SpecTpSyncSite=N(DFLASH_ACCEPT_GREEDY=7))
src=(H/"tree_dispatch.py").read_text(encoding="utf-8-sig")
for f in ast.parse(src).body:
    if isinstance(f,ast.FunctionDef):exec(compile(ast.unparse(f),str(H/"tree_dispatch.py"),"exec"),scope)
choose=scope["select_tree"];accept=scope["accept_tree"]
D=np.load(K/"public_pair/trace_08.npz")
ids=torch.from_numpy(D["ids"][:3]).long();scores=torch.from_numpy(D["scores"][:3]).float()
anchors=torch.tensor(D["tokens"][D["anchors"][:3]]).long()
table=torch.arange(1,1+6*64,dtype=torch.int32).view(6,64)
req=torch.tensor([1,4,2])
prefix=torch.tensor([12,16,21],dtype=torch.int32)
slots=table[req[:,None],prefix[:,None]+torch.arange(8)].long()
backend=type("FlashAttentionBackend",(),{})()
for k,v in dict(fa_impl_ver=3,page_size=1,use_mla=False,fa_skip_kv_cache=False,kv_cache_is_mxfp8=False,
    has_local_attention=False,use_sliding_window_kv_pool=True,sliding_window_size=5).items():
    setattr(backend,k,v)
model=type("MuseSparkTextForCausalLM",(),{})()
tr=N(defer_read_translate=False,is_translating=False)
mapping=torch.randperm(512,dtype=torch.int32)
runner=N(attn_backend=backend,model=model,kv_index_translator=tr,req_to_token_pool=N(req_to_token=table),
         token_to_kv_pool=N(full_to_swa_index_mapping=mapping))
slots_seen=[]
w=N(_draft_sampler=N(tree_scores=scores,candidate_out=ids),model_runner=runner,lilicorr=None,_need_mamba_verify_commit=False,
    _can_skip_verify_logits=lambda b,s: not b.has_grammar and not b.return_logprob)
b=N(forward_mode=N(is_idle=lambda:False),has_grammar=False,return_logprob=False,sampling_info=N(is_all_greedy=True),
    reqs=[N(kv=N(kv_allocated_len=int(l)+8)) for l in prefix],req_pool_indices=req)
def active():return choose(w,b,prefix,slots,anchors,True)
step=active();assert step is not None and step.swa.shape==(24,6) and step.full.shape==(24,32)
wire=step.wire.clone()
calls=[]
def restore(data,src):
    assert src==0
    data.copy_(wire)
    calls.append(data.numel()*data.element_size())
par.tp_group=N(world_size=8,broadcast=restore)
# A locally incorrect rank proposal must become the shared rank-zero proposal before staging.
saved=w._draft_sampler.candidate_out
w._draft_sampler.candidate_out=saved+123
repaired=active()
assert torch.equal(repaired.wire,wire)
assert torch.equal(repaired.tokens,wire[:2*3*8].view(torch.int64).view(3,8))
w._draft_sampler.candidate_out=saved
assert calls==[480]
par.tp_group=N(world_size=1)
base_ptr=w._fixed_tree_slab.full.data_ptr()
backend._kv_shard_pool=None;runner.token_to_kv_pool.swa_req_ring_size=None
checks=0
for obj,attr,value in [(backend,"page_size",16),(backend,"fa_impl_ver",4),(backend,"use_mla",True),
        (backend,"_kv_shard_pool",object()),(runner.token_to_kv_pool,"swa_req_ring_size",2048),(backend,"has_local_attention",True),(backend,"kv_cache_is_mxfp8",True),(tr,"defer_read_translate",True),
        (w,"lilicorr",1),(w,"_need_mamba_verify_commit",True),(b,"has_grammar",True),(b,"return_logprob",True),
        (b.sampling_info,"is_all_greedy",False),(par,"attn_dcp_size",2),(par,"attn_dp_enabled",True),
        (par,"enable_prefill_cp",True),(runner,"model",N()),(w._draft_sampler,"tree_scores",None)]:
    old=getattr(obj,attr);setattr(obj,attr,value)
    assert active() is None,attr
    setattr(obj,attr,old);checks+=1
assert choose(w,b,prefix,slots,anchors,False) is None;checks+=1
scope['SIMULATE_ACC_LEN']=4
assert active() is None;checks+=1
scope['SIMULATE_ACC_LEN']=0
w._tp_sync=N(enabled=lambda site:True,sync=lambda site,preds:slots_seen.append((site,preds.shape)))
rings=[]
for i in range(2):
    rings.append([torch.empty(3,dtype=torch.int32),torch.empty(3,dtype=torch.int32),torch.empty(3,dtype=torch.long),
                  torch.empty((3,8),dtype=torch.long),torch.empty(3,dtype=torch.long)])
i=iter(rings);w._next_accept_bonus_buffers=lambda bs:next(i)
pred=torch.full((3,8),-7123,dtype=torch.long)
r1=accept(w,step,pred,None,b,prefix,slots);snap=[a.clone() for a in r1[1:]]
tr.is_translating=True;tr.space=lambda kind:N(read_v2p=mapping);tr.full_v2p_table=mapping.flip(0)
step2=active();assert w._fixed_tree_slab.full.data_ptr()==base_ptr
logits=torch.randn(24,7)
r2=accept(w,step2,None,logits,b,prefix,slots)
assert all(torch.equal(a,z) for a,z in zip(r1[1:],snap))
assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(r1[1:],r2[1:]))
assert slots_seen==[(7,torch.Size([3,8]))]*2
# Native target admission fallback runs independent of require_mlp_sync. Inspect the actual packaged body.
patch=(H/"fixed_tree/candidate.diff").read_text()
for snippet in ['getattr(forward_batch.spec_info, "tree_step", None)','tree_step=tree_step,',
    "return tree.attention(self, layer, q, flash_attn_with_kvcache, out)",
    "torch.cuda.Event()","_next_accept_bonus_buffers(step.batch)","_tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY"]:
    assert snippet in patch
memory={}
for label,bb,cap in [("B1_2048",1,4096),("B4_8192",4,16384),("B16_74160",16,131072),
                    ("B32_74160",32,131072)]:
    arena=TreeStep(bb,cap,"meta",2047)
    sizes={k:v.numel()*v.element_size() for k,v in vars(arena).items() if torch.is_tensor(v) and k not in ("tokens","parents","depth","mask")}
    memory[label]=dict(bytes_per_rank=sum(sizes.values()),components=sizes)
r=dict(tp_rank_repaired_wire_bytes=calls[0],guarded_fallbacks=checks,retained_output_slots=2,tp_sync_calls=2,
    correct_static_or_unified_translations=2,small_simulation_slab_checked=True,
    tensor_storage=memory,identity_map_note="Only static full=virtual pools need an identity vector. One 1,893,609-entry int32 vector adds 7,574,436 bytes; unified tables are borrowed.",
    fixed_shape_selector_capture_extra_bytes_at_max_batch_32=32*7*16*16*4)
(H/"fixed_tree/dispatch_cpu.json").write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!="tensor_storage"}))

