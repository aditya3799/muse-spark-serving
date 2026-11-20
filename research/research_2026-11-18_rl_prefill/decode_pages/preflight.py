"""Check scored replay routing, fixed q-width, and the FA3 CUDA page selection."""
import ast,hashlib,json,re,subprocess,tempfile,os,random
from pathlib import Path
from types import SimpleNamespace as S
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def extract(path,cls,names):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'));return {f.name:f for c in tree.body if getattr(c,'name',None)==cls for f in c.body if getattr(f,'name',None) in names}
f=extract(H/'flashattention_backend.py','FlashAttentionBackend',['_set_decode_page_metadata','_apply_cuda_graph_metadata','__init__'])
def builds(major,page,typ,fa,is_draft):
    assignment=next(n for n in ast.walk(f['__init__']) if isinstance(n,ast.Assign) and any('self._muse_fa3_window_pages'==ast.unparse(t) for t in n.targets))
    fake='fp8';b=S(kv_cache_dtype=fake,page_size=page,device='fake');ns=dict(self=b,fa_impl_ver=fa,model_runner=S(is_draft_worker=is_draft,model_config=S(hf_config=S(model_type=typ))),torch=S(float8_e4m3fn=fake,cuda=S(get_device_capability=lambda _: (major,0))))
    exec(compile(ast.Module([assignment],[]),'<target activation>','exec'),ns);return b._muse_fa3_window_pages
assert builds(9,1,'muse_spark',3,False) and builds(9,1,'muse_spark_text',3,False)
for x in ((8,1,'muse_spark',3,False),(9,16,'muse_spark',3,False),(9,1,'muse_spark',4,False),(9,1,'muse_spark',3,True),(9,1,'other',3,False)):assert not builds(*x)
calls=[]
def record(*a,**kw):calls.append(kw)
ns=dict(normal_decode_set_metadata=record)
exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+[f['_set_decode_page_metadata']],[]),'<pinned FA3 replay>', 'exec'),ns)
fn=ns['_set_decode_page_metadata']
b=S(kv_index_translator=S(reads_are_translated=False),_muse_fa3_window_pages=True,use_sliding_window_kv_pool=True,topk=1,_kv_shard_pool=None,local_attn_builder=None,is_prefill_aware_swa=False,sliding_window_size=2047,page_size=1,req_to_token=object(),max_num_pages=131072,token_to_kv_pool=S(full_to_swa_index_mapping=object()))
m=S(cache_seqlens_int32=object(),cu_seqlens_k=object(),page_table=object(),swa_page_table=object())
for q in (1,3,8,13):
    fn(b,None,m,object(),object(),0 if q==1 else q,window_query_length=q)
    assert calls[-1]['swa_suffix_tokens']==2047+q
for key,val in (('topk',2),('_muse_fa3_window_pages',False),('use_sliding_window_kv_pool',False),('_kv_shard_pool',object()),('local_attn_builder',object()),('is_prefill_aware_swa',True),('sliding_window_size',0)):
    orig=getattr(b,key);setattr(b,key,val);fn(b,None,m,object(),object(),0,window_query_length=8);assert calls[-1]['swa_suffix_tokens']==0;setattr(b,key,orig)
fn(b,None,m,object(),object(),0);assert calls[-1]['swa_suffix_tokens']==0
routes=[n for n in ast.walk(f['_apply_cuda_graph_metadata']) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self._set_decode_page_metadata']
actual=sorted(ast.unparse(n.keywords[0].value) if n.keywords else 'absent' for n in routes);assert actual==['1','absent','verify_delta if ragged_layout is None else 0']
r=random.Random(91375);visited=0
for _ in range(20000):
    seq=r.randrange(1,131073);q=r.randrange(1,min(256,seq)+1);w=r.choice((31,127,2047,4095))
    # All local Hopper query tiles begin at m_idx >= 0, even with packed GQA.
    m_idx=r.randrange(q)
    n_offset=max(0,m_idx+seq-q-w)
    first_written=max(0,(seq-q-w)//256*256)
    assert first_written<=n_offset
    visited+=1
with tempfile.TemporaryDirectory(dir=H) as tmp:
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'metadata.index'));git=lambda *a:subprocess.check_output(['git','-C',str(ROOT/'sglang'),*a],env=env)
    git('read-tree','297e54fd105e974cdf401e5b4d20253a22c16f90');git('apply','--cached','--check',str(H/'candidate.diff'))
out=dict(hopper_model_fa3_page1_fp8='active',other_configs=5,mode_widths=actual,guarded_dispatches=len(calls),sampled_paged_tile_bounds=visited,cumulative_apply_check=True,pin_fa='f89bc2306632d1ec5f97b014dded4254f5b4a907')
(H/'eligibility.json').write_text(json.dumps(out,indent=2));print(out)

