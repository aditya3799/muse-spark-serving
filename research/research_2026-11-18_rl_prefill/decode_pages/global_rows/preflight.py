"""Exercise exact production metadata ownership and FA3 argument branches on CPU."""
import ast,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace as S
import torch
H=Path(__file__).resolve().parent
tree=ast.parse((H/'flashattention_backend.py').read_text())
c=next(n for n in tree.body if getattr(n,'name',None)=='FlashAttentionBackend')
methods={n.name:n for n in c.body if isinstance(n,ast.FunctionDef)}
seen=[]
def prepare(lens,cu,fake_pt,global_table,rows,capacity,seqlens,delta,page_size,swa,**kw):
    direct=kw['direct_full_indices']
    if direct is not None:direct.copy_(rows)
    seen.append(dict(width=kw['swa_suffix_tokens'],direct=direct is not None))
args=dict(torch=torch,normal_decode_set_metadata=prepare)
exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+[methods['_set_decode_page_metadata']],[]),str(H),'exec'),args)
b=S(kv_index_translator=S(reads_are_translated=False,is_translating=False),_muse_fa3_window_pages=True,use_sliding_window_kv_pool=True,topk=1,_kv_shard_pool=None,local_attn_builder=None,is_prefill_aware_swa=False,sliding_window_size=2047,page_size=1,req_to_token=torch.randint(0,500,(11,720),dtype=torch.int32),max_num_pages=700,token_to_kv_pool=S(full_to_swa_index_mapping=object()),is_encoder_decoder=False,attn_cp_size=1)
def meta(bs):return S(cache_seqlens_int32=torch.empty(bs,dtype=torch.int32),cu_seqlens_k=torch.empty(bs+1,dtype=torch.int32),page_table=torch.empty(bs,700,dtype=torch.int32),swa_page_table=torch.ones(bs,700,dtype=torch.int32),muse_full_batch_indices=None)
binds={}
for method in ('forward_decode','forward_extend'):
    nodes=[n for n in ast.walk(methods[method]) if isinstance(n,ast.If) and 'metadata.muse_full_batch_indices is not None' in ast.unparse(n.test)]
    assert len(nodes)==1;binds[method]=compile(ast.Module(nodes,[]),'<actual FA3 row binding>','exec')
events=dict()
for bs in (1,3,8):
    m=meta(bs);ptr=None
    for new_q in (8,0,3,13):
        rows=(torch.arange(bs,dtype=torch.int64)*3+new_q)%11
        if bs>1:rows[-1]=rows[0]
        args['_set_decode_page_metadata'](b,None,m,rows,torch.full((bs,),301),0,window_query_length=new_q)
        assert seen[-1]['width']==2047+(new_q or 700)
        if ptr is not None:assert ptr==m.muse_full_batch_indices.data_ptr()
        ptr=m.muse_full_batch_indices.data_ptr()
        for method,code in binds.items():
            for window in (False,True):
                original=m.swa_page_table if window else m.page_table
                ns=dict(metadata=m,kwargs={},page_table=original,self=b,is_swa_layer=window);exec(code,ns)
                if window:assert not ns['kwargs'] and ns['page_table'] is m.swa_page_table
                else:
                    indices=ns['kwargs']['cache_batch_idx']
                    assert ns['page_table'].data_ptr()==b.req_to_token.data_ptr()
                    assert torch.equal(ns['page_table'][indices.long()],b.req_to_token[rows,:700])
class Ends:POST_REPLAY='post'
class Base:
    def shared_read_ends(self,fm):return 'default'
cls=ast.ClassDef(name='Child',bases=[ast.Name('Base',ast.Load())],keywords=[],body=[methods['shared_read_ends']],decorator_list=[])
glob={'Base':Base,'SharedReadEnds':Ends}
exec(compile(ast.fix_missing_locations(ast.Module([cls],[])),str(H),'exec'),glob)
child=glob['Child']()
for active in (True,False):
    child._muse_fa3_window_pages=active
    for kind in ('verify','decode','extend'):
        mode=S(is_decode=lambda:kind=='decode',is_target_verify=lambda:kind=='verify')
        observed=child.shared_read_ends(mode);events[f'{active}_{kind}']=observed
        assert observed==('post' if active and kind!='extend' else 'default')
r=dict(cases=48,width_no_bound_reuse='full table',same_ids_storage_across_replay=True,physical_req_table_shape=[11,720],original_FA3_cap=700,duplicate_ids_checked=True,shared_read_ends=events,sha=hashlib.sha256((H/'candidate.diff').read_bytes()).hexdigest())
(H/'dispatch.json').write_text(json.dumps(r,indent=2));print(r)

