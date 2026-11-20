"""Source-extracted target TP worker and Muse model with explicit upstream residual fixtures."""
from __future__ import annotations
import ast,sys,json
from pathlib import Path
from types import MethodType,SimpleNamespace as S
import torch
from entry import stock
H=Path(__file__).resolve().parent;R=H.parent
TP="python/sglang/srt/managers/tp_worker.py"
MU="python/sglang/srt/models/muse_spark.py"
sys.path.insert(0,str(R/"research_2026-11-18_rl"))
import bench_aux
class Mode:
    def __init__(self,flag):self.flag=flag
    def need_capture(self):return self.flag
    def is_full(self):return self.flag
CAP=S(NULL=Mode(False),FULL=Mode(True))
def extract(src,cl,meth,ns):
    a=next(i for i in ast.parse(src).body if getattr(i,'name',None)==cl)
    n=next(i for i in a.body if getattr(i,'name',None)==meth);n.decorator_list=[]
    exec(compile(ast.Module(body=[n],type_ignores=[]),meth,'exec',flags=__import__('__future__').annotations.compiler_flag),ns)
    return ns[meth]
def target(variant,state,sp=False):
    early=variant=="candidate"
    src=(H/"applied_early"/MU if early else R/"release_2026-11-18/candidate_sources"/MU).read_text(encoding="utf8")
    tsrc=(H/"applied_early"/TP).read_text(encoding="utf8") if early else stock(TP)
    pack=bench_aux.pack_function()
    def tracked_pack(aux):
        if not isinstance(aux,torch.Tensor):state["packs"]+=1
        return pack(aux)
    def gather(x,num_tokens):
        state["gathers"].append(x.shape[1]);return x[:num_tokens]
    ns=dict(torch=torch,pack_aux_hidden_states=tracked_pack,layernorm_sp=S(runs_sp=lambda mode:sp,
        sp_entry_scatter=lambda x:x,sp_exit_gather=gather),get_forward=lambda:S(set=lambda *a:None))
    forward=extract(src,"MuseSpark1xModel","forward",ns.copy())
    wrap=extract(src,"MuseSparkTextForCausalLM","forward",ns.copy())
    inner=S(embed_norm=None,residual_dtype=torch.float32,operator_dtype=torch.bfloat16,layers_to_capture=list(range(5)),norm=lambda v:v)
    def produce(ids,pos,fb,input_embeds=None):
        raw=state["make"](fb)
        inner.layers=[bench_aux.ResidualLayer(t) for t in raw]
        embed=raw[0].bfloat16()
        if state.get("captured_body"):
            fb=S(forward_mode=fb.forward_mode)
        return forward(inner,ids,pos,fb,input_embeds=embed)
    choose=extract(stock("python/sglang/srt/layers/logits_processor.py"),"LogitsProcessor","_get_hidden_states_to_store",dict(torch=torch,pack_aux_hidden_states=tracked_pack))
    def logits(ids,hidden,head,fb,aux):
        # Fixture dense logits on a small, fixed subset of output features.
        selections=torch.cumsum(torch.tensor(fb._counts,device=hidden.device),0)-1
        logits=hidden[selections,:32]
        z=choose(None,hidden,None,aux,hidden,None,None,selections,fb)
        return S(hidden_states=z,next_token_logits=logits)
    model=S(model=produce,capture_aux_hidden_states=True,lm_head=None,logits_processor=logits)
    def runner(fb,**kwargs):
        out=wrap(model,fb.input_ids,None,fb)
        if state.get("monitor"):state["logits"].append(out.next_token_logits.clone())
        return S(logits_output=out,can_run_graph=False,expert_distribution_metrics=None,routed_experts_output=None,indexer_topk_output=None)
    def init(batch,runner,capture_hidden_mode,**kw):
        return S(input_ids=state["ids"],_counts=batch.extend_lens,seq_lens=batch.seq_lens,kv_loc_plan=state["loc"],
            is_prefill_only=False,forward_mode=batch.forward_mode,capture_hidden_mode=capture_hidden_mode,apply_deprecated_skip_attn_backend_init=lambda x:None)
    ns=dict(torch=torch,ForwardBatch=S(init_new=init),get_exec=lambda:S(features=S(enable_encoder_swa_bounded_replay=False)),
        GenerationBatchResult=S,capture_pre_sample_logits=lambda *a:None)
    f=extract(tsrc,"TpModelWorker","forward_batch_generation",ns)
    worker=S(set_hicache_consumer=lambda *a:None,model_runner=S(forward=runner,sample=lambda l,fb:l.next_token_logits.argmax(-1)),
       is_dllm=lambda:False,pp_group=S(is_last_rank=True),enable_overlap=False,enable_spec=True,_maybe_finalize_elastic_cuda_graph_scale=lambda:None)
    return MethodType(f,worker)
def cpu():
    d={}
    for sp in (False,True):
        base=torch.randn((5,12,32),generator=torch.Generator().manual_seed(343811))
        for early in (False,True):
            outputs=[]
            for arm in ("baseline","candidate"):
                state=dict(make=lambda fb:base,ids=torch.arange(12),loc=torch.arange(12),packs=0,gathers=[],logits=[],monitor=True)
                call=target(arm,state,sp)
                b=S(hicache_consumer_index=0,extend_lens=[7,5],seq_lens=torch.tensor([7,5]),forward_mode=None)
                omit=arm=="candidate" and early
                args=dict(capture_hidden_mode=CAP.FULL,return_kv_loc_plan=True)
                if arm=="candidate":args["dflash_skip_prefill_aux"]=omit
                result=call(b,**args);outputs.append((result,state))
            (a,x),(b,y)=outputs
            assert torch.equal(a.next_token_ids,b.next_token_ids) and torch.equal(x["logits"][0],y["logits"][0])
            assert torch.equal(a.kv_loc_plan,b.kv_loc_plan)
            if early:assert b.logits_output.hidden_states.shape==base.shape[1:] and not y["gathers"] and not y["packs"]
            else:assert torch.equal(a.logits_output.hidden_states,b.logits_output.hidden_states)
            d[f"sp_{sp}_omit_{early}"]={"parent_packs":x["packs"],"candidate_packs":y["packs"],"parent_gathers":x["gathers"],"candidate_gathers":y["gathers"]}
    state=dict(make=lambda fb:base,ids=torch.arange(12),loc=torch.arange(12),packs=0,gathers=[],logits=[],monitor=True,captured_body=True)
    call=target("candidate",state,True)
    result=call(S(hicache_consumer_index=0,extend_lens=[7,5],seq_lens=torch.tensor([7,5]),forward_mode=None), capture_hidden_mode=CAP.FULL,return_kv_loc_plan=True,dflash_skip_prefill_aux=True)
    assert result.logits_output.hidden_states.shape==base.shape[1:] and state["packs"]==0 and state["gathers"]==[]
    assert torch.equal(result.next_token_ids,a.next_token_ids)
    d["captured_body_with_eager_outer"]={"original_five_taps_replayed":True,"outer_gathers":state["gathers"],"packs":state["packs"]}
    (H/"early_target_cpu.json").write_text(json.dumps(d,indent=2));print(d)
if __name__=="__main__":cpu()

