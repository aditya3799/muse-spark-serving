"""Exact exported helper/control tests; no CUDA or attention arithmetic proof."""
import ast, copy, hashlib, importlib.util, json
from pathlib import Path
from types import SimpleNamespace as NS
P=Path(__file__).resolve().parent
REL='python/sglang/srt/layers/attention/'
HELPER=P/'candidate_sources'/REL/'fa3_swa_scheduler.py'
BACKEND=P/'candidate_sources'/REL/'flashattention_backend.py'
BASE=P.parents[2]/'coordination/round5_humming_adaptive_fused/candidate_sources'/REL/'flashattention_backend.py'
spec=importlib.util.spec_from_file_location('exact_swa_helper',HELPER)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
class Tensor:
    def __init__(self,shape,values=None):self.shape=shape;self.values=values or [];self.dtype='bf16';self.device='cuda:0'
class Factory:
    def __init__(self):self.calls=[];self.tape=None
    def __call__(self,batch_size,max_seqlen_q,max_seqlen_k,num_heads,num_heads_k,headdim,headdim_v,cache_seqlens,qkv_dtype,cu_seqlens_q,cu_seqlens_k_new,page_size,max_seqlen_k_new,causal,window_size,has_softcap,num_splits,pack_gqa,sm_margin):
        args=locals().copy();args.pop('self');out=NS(values=None)
        self.calls.append(args)
        def run():out.values=(tuple(cache_seqlens.values),tuple(cu_seqlens_q.values),None if cu_seqlens_k_new is None else tuple(cu_seqlens_k_new.values),window_size,max_seqlen_k,num_splits)
        if self.tape is None:run()
        else:self.tape.append(run)
        return out

def fixture(bucket=3):
    f=Factory();b=NS(_get_scheduler_metadata=f,_swa_scheduler_reuse_disabled=False,fa_impl_ver=3,use_mla=False,topk=1,_disable_scheduler_metadata_precompute=False,local_attn_builder=None,_kv_shard_pool=None,is_prefill_aware_swa=False,page_size=1,num_attention_heads=16,num_kv_heads=2,head_dim=64,has_softcap=False,num_splits=0)
    fb=NS(batch_size=3,forward_mode=NS(is_target_verify=lambda:True))
    m=NS(cache_seqlens_int32=Tensor((3,),[7,50,128]),cu_seqlens_q=Tensor((4,),[0,8,16,24]),cu_seqlens_k=Tensor((4,),[0,7,57,185]),page_table=Tensor((3,4096)),swa_page_table=Tensor((3,4096)),image_token_ranges=None,swa_spec_metadata=None)
    layer=NS(tp_q_head_num=16,tp_k_head_num=2,head_dim=64,v_head_dim=64,logit_cap=0,is_cross_attention=False)
    kw=dict(backend=b,forward_batch=fb,metadata=m,layer=layer,q=Tensor((24,16,64)),key_cache=Tensor((8192,1,2,64)),value_cache=Tensor((8192,1,2,64)),page_table=m.swa_page_table,cache_seqlens=m.cache_seqlens_int32,cu_seqlens_q=m.cu_seqlens_q,cu_seqlens_k_new=m.cu_seqlens_k,max_seqlen_q=8,window_size=(2047,0),causal=True,cp_active=False,use_local_attn=False,use_cascade_attn=False,sinks=None,score_mod=None,rel_bias=None)
    fb.batch_size=bucket
    m.cache_seqlens_int32.shape=(bucket,)
    m.cache_seqlens_int32.values=[7+41*i for i in range(bucket)]
    m.cu_seqlens_q.shape=(bucket+1,)
    m.cu_seqlens_q.values=[8*i for i in range(bucket+1)]
    m.cu_seqlens_k.shape=(bucket+1,)
    m.cu_seqlens_k.values=[sum(m.cache_seqlens_int32.values[:i]) for i in range(bucket+1)]
    m.page_table.shape=m.swa_page_table.shape=(bucket,4096)
    kw['q'].shape=(bucket*8,16,64)
    return f,b,fb,m,layer,kw

def run():
    f,b,fb,m,layer,kw=fixture();assert h.supports_swa_scheduler_reuse(f)
    assert not h.supports_swa_scheduler_reuse(None)
    assert not h.supports_swa_scheduler_reuse(lambda **kwargs:None)
    assert h.get_swa_verify_scheduler_metadata(**kw) is None
    h.reset_swa_scheduler_cache(b,fb,allow=True)
    a=h.get_swa_verify_scheduler_metadata(**kw)
    for _ in range(62):assert h.get_swa_verify_scheduler_metadata(**kw) is a
    assert len(f.calls)==1
    kw['window_size']=(4095,0);other=h.get_swa_verify_scheduler_metadata(**kw)
    assert other is not a and len(f.calls)==2
    for name in ('max_seqlen_k_new','sm_margin'):assert all(c[name]==0 for c in f.calls)
    assert all(c['pack_gqa'] is None and c['causal'] is True for c in f.calls)
    assert f.calls[0]['max_seqlen_k']==4096 and f.calls[0]['cu_seqlens_k_new'] is None
    m.cache_seqlens_int32.values[:]=[9,60,150]
    h.reset_swa_scheduler_cache(b,fb,allow=True)
    fresh=h.get_swa_verify_scheduler_metadata(**kw);assert fresh is not other and fresh.values[0]==(9,60,150)
    # Separate warmups and captures must each create a fresh per-graph operation.
    graphs=[]
    for bucket in (3,7,3):
        f,b,fb,m,layer,kw=fixture(bucket)
        for _ in range(2):
            h.reset_swa_scheduler_cache(b,fb,allow=True);h.get_swa_verify_scheduler_metadata(**kw)
        h.reset_swa_scheduler_cache(b,fb,allow=False)
        assert h.get_swa_verify_scheduler_metadata(**kw) is None
        f.tape=[];h.reset_swa_scheduler_cache(b,fb,allow=True)
        output=h.get_swa_verify_scheduler_metadata(**kw)
        for _ in range(10):assert h.get_swa_verify_scheduler_metadata(**kw) is output
        assert len(f.tape)==1
        graphs.append((f.tape,output,m,f));f.tape=None
    calls=[len(factory.calls) for _,_,_,factory in graphs]
    for step in range(12):
        for index,(tape,output,meta,factory) in enumerate(graphs):
            meta.cache_seqlens_int32.values[:]=[step+index*30+i*100 for i in range(meta.cache_seqlens_int32.shape[0])]
            for op in tape:op()
            assert output.values[0]==tuple(meta.cache_seqlens_int32.values)
            assert len(output.values[0])==meta.cache_seqlens_int32.shape[0]
            assert len(output.values[1])==meta.cu_seqlens_q.shape[0]
    assert [len(factory.calls) for _,_,_,factory in graphs]==calls
    assert len({id(o) for _,o,_,_ in graphs})==3
    # Different geometry in the same forward must never consume another plan.
    changes=[('window_size',(15,0))]
    positive=0
    for name,value in changes:
        f,b,fb,m,layer,kw=fixture();h.reset_swa_scheduler_cache(b,fb,allow=True)
        one=h.get_swa_verify_scheduler_metadata(**kw);kw[name]=value
        two=h.get_swa_verify_scheduler_metadata(**kw);assert one is not two and len(f.calls)==2;positive+=1
    f,b,fb,m,layer,kw=fixture();h.reset_swa_scheduler_cache(b,fb,allow=True)
    one=h.get_swa_verify_scheduler_metadata(**kw)
    kw['max_seqlen_q']=16;kw['q'].shape=(48,16,64)
    two=h.get_swa_verify_scheduler_metadata(**kw);assert one is not two;positive+=1
    for name,value in [('num_splits',1),('num_splits',8)]:
        f,b,fb,m,layer,kw=fixture();h.reset_swa_scheduler_cache(b,fb,allow=True)
        one=h.get_swa_verify_scheduler_metadata(**kw);setattr(b,name,value)
        two=h.get_swa_verify_scheduler_metadata(**kw);assert one is not two and f.calls[-1][name]==value;positive+=1
    guards=[('b','fa_impl_ver',4),('b','use_mla',True),('b','topk',2),('b','_disable_scheduler_metadata_precompute',True),('b','local_attn_builder',object()),('b','_kv_shard_pool',object()),('b','is_prefill_aware_swa',True),('b','page_size',16),('b','num_attention_heads',8),('b','num_kv_heads',1),('b','head_dim',128),('b','has_softcap',True),('layer','is_cross_attention',True),('layer','v_head_dim',128),('m','image_token_ranges',object()),('m','swa_spec_metadata',object()),('kw','cp_active',True),('kw','use_local_attn',True),('kw','use_cascade_attn',True),('kw','causal',False),('kw','window_size',(-1,-1)),('kw','window_size',(2047,10)),('kw','sinks',object()),('kw','score_mod',object()),('kw','rel_bias',object()),('kw','page_table',Tensor((3,4096))),('kw','cache_seqlens',Tensor((3,))),('kw','cu_seqlens_q',Tensor((4,))),('kw','cu_seqlens_k_new',Tensor((4,)))]
    for owner,name,value in guards:
        f,b,fb,m,layer,kw=fixture();h.reset_swa_scheduler_cache(b,fb,allow=True)
        if owner=='kw':kw[name]=value
        else:setattr({'b':b,'m':m,'layer':layer}[owner],name,value)
        assert h.get_swa_verify_scheduler_metadata(**kw) is None,(owner,name)
        assert len(f.calls)==0
    for tensor_name,bad_shape in [('page_table',(2,4096)),('cache_seqlens',(2,)),('cu_seqlens_q',(3,)),('q',(23,16,64))]:
        f,b,fb,m,layer,kw=fixture();h.reset_swa_scheduler_cache(b,fb,allow=True)
        kw[tensor_name].shape=bad_shape
        assert h.get_swa_verify_scheduler_metadata(**kw) is None and not f.calls
    for which in ('q','key_cache','value_cache'):
        f,b,fb,m,layer,kw=fixture();h.reset_swa_scheduler_cache(b,fb,allow=True)
        kw[which].dtype='fp8';assert h.get_swa_verify_scheduler_metadata(**kw) is None
    f,b,fb,m,layer,kw=fixture();b._swa_scheduler_reuse_disabled=True
    h.reset_swa_scheduler_cache(b,fb,allow=True);assert h.get_swa_verify_scheduler_metadata(**kw) is None
    b._swa_scheduler_reuse_disabled=False;fb.forward_mode.is_target_verify=lambda:False
    h.reset_swa_scheduler_cache(b,fb,allow=True);assert h.get_swa_verify_scheduler_metadata(**kw) is None
    # Verify reset hook appears before any branch in exact backend source.
    def methods(path):
        cls=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='FlashAttentionBackend')
        return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    old,new=methods(BASE),methods(BACKEND)
    hooks={'init_forward_metadata':True,'init_forward_metadata_in_graph':True,'init_forward_metadata_out_graph':False}
    for name,allow in hooks.items():
        body=new[name].body
        if isinstance(body[0],ast.Expr) and isinstance(body[0].value,(ast.Str,ast.Constant)):body=body[1:]
        assert isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Call)
        call=body[0].value;assert isinstance(call.func,ast.Name) and call.func.id=='reset_swa_scheduler_cache'
        assert next(k.value.value for k in call.keywords if k.arg=='allow')==allow
    # Removing only new helper dispatch restores entire scored extend AST.
    extended=copy.deepcopy(new['forward_extend'])
    class Remove(ast.NodeTransformer):
        def visit_If(self,node):
            if any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='get_swa_verify_scheduler_metadata' for c in ast.walk(node)) and len(node.body)==1 and isinstance(node.body[0],ast.Assign):return None
            return self.generic_visit(node)
    extended=Remove().visit(extended)
    assert ast.dump(extended)==ast.dump(old['forward_extend'])
    return {'passed':True,'same_geometry_layer_reuse_checks':62,'separate_graph_recordings':3,'changed_length_graph_replays':36,'geometry_partition_checks':positive,'negative_guard_cases':len(guards)+9,'exact_backend_lifecycle_hooks':3,'scored_forward_extend_unchanged_except_helper_dispatch':True,'source_hashes':{str(x.relative_to(P)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (HELPER,BACKEND)},'scope':'Exact exported Python helper/AST and simulated scheduler device recording only; no real CUDA/FA3 arithmetic, semaphore lifecycle, kernel output, or performance proof'}
if __name__=='__main__':
    result=run();(P/'cpu_lifecycle_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))