import ast, hashlib, inspect, json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
from test_candidate import backend_fixture, SOURCE, ROOT, Mode, RaggedVerifyLayout

records=[]
sig=inspect.signature(flash_attn_with_kvcache)
def actual_flash(**kw):
    # The backend's repository wrapper consumes `ver`; the installed AOT
    # entry can omit that dispatch-only argument in older packages.
    if 'ver' not in sig.parameters: kw.pop('ver',None)
    if 'out' not in sig.parameters and kw.get('out') is None: kw.pop('out',None)
    return flash_attn_with_kvcache(**kw)

def configure(b, page, length, metadata):
    b.page_size=page
    b.max_context_len=length
    b._get_scheduler_metadata=get_scheduler_metadata
    b.fa_skip_kv_cache=False
    b._mxfp8_sf_kwargs=lambda *args:{}
    b.prepare_paged_mha_query=lambda q,qr,kr,layer,**kw:(q,qr,kr,None,None)
    b.forward_extend.__globals__['flash_attn_with_kvcache']=actual_flash
    b._verify_sched_meta_buf=torch.zeros(1+8*4,dtype=torch.int32,device='cuda')
    b.target_verify_metadata=metadata

def run():
    torch.manual_seed(19117)
    for page in (1,256):
        length=8192
        b,_,_,_=backend_fixture('cuda')
        configure(b,page,length,{})
        cases={}
        for bs in (1,4,8):
            lens=torch.full((bs,),17,dtype=torch.int32,device='cuda')
            m=NS(cache_seqlens_int32=torch.zeros_like(lens),cu_seqlens_q=torch.arange(0,(bs+1)*8,8,dtype=torch.int32,device='cuda'),cu_seqlens_k=torch.zeros(bs+1,dtype=torch.int32,device='cuda'),max_seq_len_q=8,max_seq_len_k=length,scheduler_metadata=None,page_table=torch.arange(bs*length//page,dtype=torch.int32,device='cuda').view(bs,-1),image_token_ranges=None,swa_spec_metadata=None)
            b.target_verify_metadata[bs]=m
            fb=NS(batch_size=bs,req_pool_indices=torch.arange(bs,dtype=torch.int32,device='cuda'),seq_lens=lens,seq_lens_cpu=None,encoder_lens=None,forward_mode=Mode(),spec_info=NS(),positions=torch.arange(bs*8,device='cuda'),kv_loc_plan=None,out_cache_loc=None)
            q=torch.randn(bs*8,16,64,dtype=torch.bfloat16,device='cuda')
            k=torch.randn(bs*length//page,page,2,64,dtype=q.dtype,device='cuda')
            v=torch.randn_like(k)
            b.get_paged_mha_kv_cache=lambda layer,k=k,v=v:(k,v)
            layer=NS(sliding_window_size=-1,is_cross_attention=False,attn_type=1,tp_k_head_num=2,tp_v_head_num=2,tp_q_head_num=16,head_dim=64,v_head_dim=64,scaling=0.125,logit_cap=0)
            b._init_full_cg_decode_metadata(fb,True)
            for _ in range(3): b.forward_extend(q,None,None,layer,fb,save_kv_cache=False)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result=b.forward_extend(q,None,None,layer,fb,save_kv_cache=False)
            cases[bs]=(m,fb,q,k,v,layer,graph,result)
        for bs in (1,4,8,4,1,8):
            m,fb,q,k,v,layer,graph,result=cases[bs]
            for newlen in (9,513,4096,8180):
                fb.seq_lens.fill_(newlen)
                if bs>1: fb.seq_lens[0]=1
                q.normal_()
                b._init_full_cg_decode_metadata(fb,False)
                assert m.scheduler_metadata.data_ptr()==b._verify_sched_meta_buf.data_ptr()
                graph.replay()
                ref=actual_flash(q=q,k_cache=k,v_cache=v,page_table=m.page_table,cache_seqlens=m.cache_seqlens_int32,cu_seqlens_q=m.cu_seqlens_q,cu_seqlens_k_new=m.cu_seqlens_k,max_seqlen_q=8,causal=True,softmax_scale=0.125)
                assert torch.equal(result,ref.reshape(bs*8,-1)), (page,bs,newlen,(result-ref.reshape(bs*8,-1)).abs().max().item())
                records.append(dict(page=page,batch=bs,length=newlen,scheduler_elements=m.scheduler_metadata.numel(),equal=True))
            # Exercise real ragged padding/geometry on the same captured
            # query upper bound. Only the graph's live varlen rows are compared.
            verify_lens=[3] if bs==1 else ([3,4,5] if bs==4 else list(range(1,bs)))
            fb.spec_info.ragged_verify_layout=RaggedVerifyLayout.from_verify_lens(verify_lens_cpu=verify_lens,device='cuda',grid=[bs*4])
            b._init_full_cg_decode_metadata(fb,False)
            assert int((m.cu_seqlens_q[1:]-m.cu_seqlens_q[:-1]).max()) <= m.max_seq_len_q
            graph.replay()
            ref=actual_flash(q=q,k_cache=k,v_cache=v,page_table=m.page_table,cache_seqlens=m.cache_seqlens_int32,cu_seqlens_q=m.cu_seqlens_q,cu_seqlens_k_new=m.cu_seqlens_k,max_seqlen_q=8,causal=True,softmax_scale=0.125)
            assert torch.equal(result[:bs*4],ref.reshape(bs*8,-1)[:bs*4]), dict(page=page,batch=bs,max_abs=float((result[:bs*4]-ref.reshape(bs*8,-1)[:bs*4]).abs().max()))
            records.append(dict(page=page,batch=bs,ragged=True,scheduler_elements=m.scheduler_metadata.numel(),equal=True))
            del fb.spec_info.ragged_verify_layout
            # Uniform/ragged captures normally use distinct keys. Restore the
            # uniform query view before revisiting this synthetic shared key.
            m.cu_seqlens_q.copy_(torch.arange(0,(bs+1)*8,8,dtype=torch.int32,device='cuda'))
        del cases,b,q,k,v,graph,result,ref
    return dict(passed=True,device=torch.cuda.get_device_name(),candidate_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),records=records,limitations='BF16 only; exact exported scheduling, capture/replay and forward methods with mocked page-table builder (length fill); no model or multi-GPU/Hopper FP8')

try:
    result=run()
except Exception as exc:
    result=dict(passed=False,exception_type=type(exc).__name__,error=str(exc),records=records)
    import traceback; traceback.print_exc()
(ROOT/'candidate_gpu_tests.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
print('cases',len(records),flush=True)
