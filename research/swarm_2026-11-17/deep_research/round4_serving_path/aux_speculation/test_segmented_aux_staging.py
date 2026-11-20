"""Actual extracted append/FC/KV/CPU RMS flow; reduced geometry, fake RoPE/writer."""
from __future__ import annotations
import ast,contextlib,hashlib,itertools,json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Optional,Tuple,Union,Any
import torch
from torch import nn
from torch.nn import functional as F
import prefix_width_scaffold as loader
P=Path(__file__).parent;ROOT=P.parents[3];SRT=ROOT/'sglang/python/sglang/srt'
torch.set_num_threads(1)
def node(path,name,cls=None):
    t=ast.parse(path.read_text());body=t.body if cls is None else next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    return next(n for n in body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
class UnquantizedLinearMethod:pass
ns=dict(globals(),draft_tp_context=lambda _:contextlib.nullcontext(),_is_npu=False,
        logger=N(warning=lambda *a:None))
paths={'model':SRT/'models/dflash.py','worker':SRT/'speculative/dflash_worker_v2.py','norm':SRT/'layers/layernorm.py','utils':SRT/'speculative/dflash_utils.py'}
loader.load([node(paths['utils'],'can_dflash_slice_qkv_weight'),node(paths['norm'],'forward_native','RMSNorm')],ns)
Norm=type('ExtractedNativeRMS',(),{'__call__':ns['forward_native']})
for cls,names in [('DFlashDraftModel',['project_target_hidden','prepare_context_hidden_for_kv']),('DFlashAttention',['kv_proj_only','apply_k_norm','apply_k_rope'])]:
    loader.load([node(paths['model'],name,cls) for name in names],ns)
loader.load([node(paths['worker'],'_append_target_hidden_to_draft_kv_by_loc','DFlashWorkerV2')],ns)
Model=type('ExtractedDraft',(),{n:ns[n] for n in ('project_target_hidden','prepare_context_hidden_for_kv')})
Attn=type('ExtractedAttention',(),{n:ns[n] for n in ('kv_proj_only','apply_k_norm','apply_k_rope')})
Worker=type('ExtractedWorker',(),{'append':ns['_append_target_hidden_to_draft_kv_by_loc']})
def norm(n,dtype):
    r=Norm();r.hidden_size=n;r.override_orig_dtype=None;r.variance_size_override=None
    r.variance_epsilon=1e-6;r.fp32_residual=False;r.cast_x_before_out_mul=False
    r.weight=torch.linspace(.5,1.5,n,dtype=dtype)
    return r

def fixture(dtype):
    torch.manual_seed(71)
    model=Model();model.quantizable=False;model.num_context_features=5;model.config=N(hidden_size=32)
    model.fc=nn.Linear(160,32,bias=False,dtype=dtype);model.hidden_norm=norm(32,dtype);model.layers=[]
    for layerid in range(5):
        a=Attn();a.q_size=32;a.kv_size=16;a.num_kv_heads=2;a.head_dim=8;a.k_norm=norm(8,dtype)
        a.qkv_proj=N(weight=torch.randn(64,32,dtype=dtype)*.1,bias=None,quant_method=UnquantizedLinearMethod())
        a.v_scale=.75;a.attn=N(layer_id=layerid,k_scale=1.,v_scale=1.)
        a.rotary_emb=lambda positions,q,k:(q,k+positions[:,None].to(k.dtype)*.001) # explicit fake RoPE
        model.layers.append(N(self_attn=a))
    calls=[]
    def writer(attn,slots,lengths,k,v,*scales):
        bs,width=slots.shape;valid=(torch.arange(width)[None,:]<lengths[:,None]).flatten()
        calls.append({'layer':attn.layer_id,'slots':slots.flatten()[valid].clone(),'k':k[valid].clone(),'v':v[valid].clone(),'input_rows':k.shape[0]})
    w=Worker();w.model_runner=N(device=torch.device('cpu'));w.draft_parallel_layout=None
    w.draft_model=model;w.lilicorr=None;w._use_fused_kv_materialize=False;w._fused_kv_helper=None
    w.draft_model_runner=N(token_to_kv_pool=N(set_kv_buffer_prefix_valid=writer))
    return w,calls

def main():
    cases=checks=0;negative=[]
    for dtype,bs,generation in itertools.product((torch.float32,torch.bfloat16),(1,3,8,32),range(8)):
        w,calls=fixture(dtype);torch.manual_seed(101+generation)
        hidden=torch.randn(bs,8,160,dtype=dtype);commit=(torch.arange(bs)+generation)%8+1
        slots=torch.arange(bs*8).view(bs,8)+1000;positions=torch.arange(bs)[:,None]*10+torch.arange(8)[None,:]
        shared_first=hidden[:,:4].clone();staging=torch.zeros_like(hidden)
        staging[:,:4].copy_(shared_first)
        assert staging.data_ptr()!=shared_first.data_ptr()
        shared_first.fill_(-91) # next target overwrites first graph output
        survivors=(commit>4).nonzero().flatten().flip(0)
        if len(survivors):staging[survivors,4:]=hidden[survivors,4:]
        # Baseline and staged append preserve the same full8B geometry.
        args=dict(cache_loc=slots.flatten(),cache_loc_2d=slots,positions=positions.flatten(),commit_lens=commit.int())
        w.append(target_hidden=hidden.flatten(0,1),**args);baseline=calls[:];calls.clear()
        w.append(target_hidden=staging.flatten(0,1),**args)
        assert len(baseline)==len(calls)==5
        for a,b in zip(baseline,calls):
            assert a['input_rows']==b['input_rows']==bs*8
            for key in ('slots','k','v'):
                assert torch.equal(a[key],b[key]), (str(dtype),bs,generation,a['layer'],key,float((a[key].float()-b[key].float()).abs().max()))
                assert torch.isfinite(b[key]).all();checks+=1
        valid=(torch.arange(8)[None,:]<commit[:,None])
        assert torch.equal(hidden[valid],staging[valid]);checks+=1;cases+=1
    # A retained view into shared output would now carry poison.
    a=torch.arange(40).reshape(8,5);view=a[:4];safe=view.clone();a.fill_(-91)
    assert not torch.equal(view,safe);negative.append('first_aux_view_without_copy_corrupts_on_second_forward')
    # A compact survivor index cannot be used as its original request index.
    original=torch.arange(3*8*5).reshape(3,8,5);rows=torch.tensor([2,0]);wrong=torch.zeros_like(original);right=torch.zeros_like(original)
    right[rows,4:]=original[rows,4:];wrong[:2,4:]=original[rows,4:]
    assert not torch.equal(right,wrong);negative.append('compacted_aux_scatter_without_request_map_wrong')
    report={'status':'CPU_SOURCE_APPEND_STAGING_PASS','cases':cases,'exact_finite_checks':checks,'negative_controls':negative,
      'actual_extracted_methods':['DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc','DFlashDraftModel.project_target_hidden','prepare_context_hidden_for_kv','DFlashAttention.kv_proj_only','apply_k_norm','apply_k_rope','RMSNorm.forward_native','can_dflash_slice_qkv_weight'],
      'scope':'CPU FP32/BF16,5layers, context160→hidden32,KVheads2×8; source control and nativeCPU arithmetic only',
      'fakes':['Target hidden rows are synthetic','RoPE delegate replaced by deterministic rowwise position transform','KV pool writer records valid prefixes rather than quantizing/cache storing','Draft TP context no-op; no distributed collectives','NativeCPU RMS backend, not actual CUDA fused RMS'],
      'production_staging_bytes_B32_BF16':32*8*40960*2,
      'full_append_compute_retained':True,'limits':['Actual40960→8192 GPU FC and nativeKV kernels nottested','Targetsegmentedforward arithmetic unknown','No performance or serving claim'],
      'sources':{str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in paths.values()},'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (P/'SEGMENTED_AUX_STAGING_CPU.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()