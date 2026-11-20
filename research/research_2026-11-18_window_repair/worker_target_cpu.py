"""Verify the real DFlash EXTEND→TP→Muse→logits chain on CPU, including fallbacks."""
import json
from array import array
from pathlib import Path
from types import SimpleNamespace as S,MethodType
import torch
from entry import WORK,MODE,stock
from target_entry import CAP,target,extract
from regions import DFlashRegionCoverage
H=Path(__file__).resolve().parent
def req(n):
    ids=array("q",range(n))
    return S(full_untruncated_fill_ids=ids,cache_salt=None,extra_key=None,session=None,multimodal_inputs=None,positional_embed_overrides=None,skip_radix_cache_insert=False)
def main():
    rows={}
    for prefixes,lengths,ends,client in [([16],[16],[82],False),([64],[16],[82],False),([16,0],[16,4],[82,4],False),([16],[16],[82],True)]:
        a=next(iter(lengths));m=sum(lengths);raw=torch.randn(5,m,32,generator=torch.Generator().manual_seed(1753))
        states=[]
        for arm in ("baseline","candidate"):
            cov=DFlashRegionCoverage(8,1)
            st=dict(make=lambda fb:raw,ids=torch.arange(m),loc=torch.arange(m),packs=0,gathers=[],logits=[],monitor=True,draft=[])
            cb=target(arm,st,True)
            pos=torch.cat([torch.arange(p,p+n) for p,n in zip(prefixes,lengths)])
            ns=dict(torch=torch,CaptureHiddenMode=CAP,ForwardMode=S(EXTEND=MODE),SpecTpSyncSite=S(DFLASH_TARGET=True),compute_position=lambda *args:(pos,None),async_h2d=lambda x,**kw:torch.tensor(x,**kw))
            worker=S(_validate_phase1_sampling_support=lambda *a:None,_wait_for_fused_hicache_load=lambda *a:None,
                _tp_sync=S(sync=lambda *a:None),target_worker=S(forward_batch_generation=cb),model_runner=S(prefill_attention_backend_str="fa3"),
                _draft_write_ids=lambda plan:plan,_make_next_draft_input_prefill=lambda **kw:kw,
                _append_target_hidden_to_draft_kv_by_loc=lambda **kw:st["draft"].append(kw["target_hidden"].clone()))
            if arm=="candidate":worker._window_coverage=cov;worker._can_omit_target_aux=True
            b=S(forward_mode=MODE,reqs=[req(n) for n in ends],is_extend_in_batch=False,extend_lens=lengths,prefix_lens=prefixes,
                seq_lens=torch.tensor([p+n for p,n in zip(prefixes,lengths)]),out_cache_loc=st["loc"],return_hidden_states_mode=CAP.FULL if client else CAP.NULL,hicache_consumer_index=0)
            src=stock(WORK) if arm=="baseline" else (H/"applied_early"/WORK).read_text()
            res=extract(src,"DFlashWorkerV2","forward_batch_generation",ns)(worker,b)
            states.append((st,res))
        (s0,b),(s1,c)=states
        assert torch.equal(b.next_token_ids,c.next_token_ids) and torch.equal(s0["logits"][0],s1["logits"][0])
        assert torch.equal(b.next_draft_input["seq_lens"],c.next_draft_input["seq_lens"])
        skip=prefixes==[16] and lengths==[16]
        assert len(s1["draft"])==int(not skip)
        if not skip:assert torch.equal(s0["draft"][0],s1["draft"][0])
        assert s1["packs"]==int(client or not skip) and len(s1["gathers"])==s1["packs"]
        rows[str((prefixes,lengths,client))]=dict(draft_materialized=bool(s1["draft"]),packs=s1["packs"],gathers=s1["gathers"],same_logits=True)
    (H/"early_worker_cpu.json").write_text(json.dumps(rows,indent=2));print(rows)
if __name__=="__main__":main()

