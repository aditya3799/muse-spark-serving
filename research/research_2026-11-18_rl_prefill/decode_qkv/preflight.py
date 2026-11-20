"""Qualify source-selected FA3 write-buffer ownership without allocating a GPU."""
import ast,hashlib,json,subprocess,re
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def methods(path,class_name,names,scope):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    body=next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    keep=[n for n in body if getattr(n,'name',None) in names]
    assert len(keep)==len(names)
    d=dict(torch=torch,**scope)
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+keep,[]),str(path),'exec'),d)
    return {k:d[k] for k in names}
mem=ROOT/'sglang/python/sglang/srt/mem_cache/memory_pool.py'
tree=ast.parse(mem.read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KVWriteLoc')
from dataclasses import dataclass
d={'torch':torch,'dataclass':dataclass,'Optional':__import__('typing').Optional}
exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+[node],[]),str(mem),'exec'),d)
methods_=methods(HERE/'flashattention_backend.py','FlashAttentionBackend',['_mha_fused_write_buffers','muse_fp8_write_buffers','get_fused_kv_write_buffers'],dict(KVWriteLoc=d['KVWriteLoc'],is_cp_active=lambda fb:fb.cp))
kv=torch.zeros(37,2,64,dtype=torch.uint8).view(torch.float8_e4m3fn); vv=kv.clone()
loc=torch.tensor([3,7,0,32]);swa=torch.tensor([15,4,0,28])
fb=SimpleNamespace(out_cache_loc=loc,out_cache_loc_is_physical=False,dcp_kv_mask=None,encoder_out_cache_loc=None,cp=False)
b=SimpleNamespace(kv_cache_dtype=torch.float8_e4m3fn,_muse_cast_map=torch.empty(65536,dtype=torch.uint8),topk=1,kv_cache_is_mxfp8=False,use_mla=False,kv_index_translator=SimpleNamespace(is_translating=False),_kv_shard_pool=None,fa_skip_kv_cache=False,forward_metadata=SimpleNamespace(swa_out_cache_loc=swa),use_sliding_window_kv_pool=True,token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda _: (kv,vv)))
for k,v in methods_.items():setattr(b,k,MethodType(v,b))
l=SimpleNamespace(k_scale=None,v_scale=None,head_dim=64,v_head_dim=64,is_cross_attention=False,tp_k_head_num=2,tp_v_head_num=2,layer_id=7,sliding_window_size=-1)
for window in (-1,2047):
    l.sliding_window_size=window
    answer=b.muse_fp8_write_buffers(l,fb)
    assert answer[0].data_ptr()==kv.data_ptr() and answer[1].data_ptr()==vv.data_ptr()
    assert answer[2] is (swa if window>0 else loc) and answer[3] is b._muse_cast_map
    assert b.get_fused_kv_write_buffers(l,fb) is None
guards=[(b,'_muse_cast_map',None),(b,'topk',2),(l,'k_scale',1.),(l,'v_scale',1.),(b,'use_mla',True),(l,'is_cross_attention',True),(b.kv_index_translator,'is_translating',True),(b,'_kv_shard_pool',object()),(b,'fa_skip_kv_cache',True),(l,'tp_k_head_num',1),(fb,'cp',True),(fb,'dcp_kv_mask',torch.ones(4)),(b,'kv_cache_is_mxfp8',True),(b.forward_metadata,'swa_out_cache_loc',None)]
for obj,key,value in guards:
    original=getattr(obj,key);setattr(obj,key,value)
    assert b.muse_fp8_write_buffers(l,fb) is None,key
    setattr(obj,key,original)
bf=torch.arange(65536).short().view(torch.bfloat16)
table=bf.to(torch.float8_e4m3fn).view(torch.uint8)
perm=torch.randperm(65536,generator=torch.Generator().manual_seed(8411))
assert torch.equal(table[perm],bf[perm].to(torch.float8_e4m3fn).view(torch.uint8))
status=json.loads((ROOT/'release_2026-11-18/scored_submission_status.json').read_text())
raw=status['result']['startup_log']['text'];raw=' '.join(raw) if isinstance(raw,list) else raw
flags={k: re.search(k+r"'\s*:\s*([^,]+)",raw)[1].strip("'") for k in ('kv_cache_dtype','page_size','enable_unified_memory')}
result=dict(scored_flags=flags,scored_verify_graph_width_8='num_tokens_per_req=8' in raw,writes=['full ids preserve pointer','translated SWA ids preserve pointer'],fallback_cases=[key for obj,key,value in guards],cast_table_all_65536_cpu='matched',candidate_sha256=hashlib.sha256((HERE/'candidate.diff').read_bytes()).hexdigest())
print(json.dumps(result,indent=2));(HERE/'eligibility.json').write_text(json.dumps(result,indent=2))

