import ast,json,contextlib
from pathlib import Path
from types import SimpleNamespace as N
import torch
from lookup_runtime import Fixed8LookupProposer,Snapshot
P=Path(__file__).resolve().parent
s=(P/'dflash_worker_lookup_overlay.py').read_text()
a=s.index('        # Isolated research hook:')
b=s.index('        # Must stay ahead of the target verify launch below.',a)
fragment=s[a:b]
source='def probe(self,batch,block_ids,prefix_lens,bs,forward_batch):\n'+fragment+'        return draft_tokens\n'
namespace={'torch':torch,'draft_tp_context':lambda _:contextlib.nullcontext(),'_is_all_greedy':lambda s:s.is_all_greedy}
exec(compile(source,'actual_worker_fragment','exec'),namespace)
checks=[]
history=tuple(list(range(1,50))+[90,91,1,2,3,4])

def run(snapshots, *, enabled=True, world=1, grammar=False, sampling=True, anchors=None, prefixes=None, proposer=True):
    bs=len(snapshots); events=[]
    sampler=N(out=torch.full((bs*7,),777,dtype=torch.int64),stage_sampling_params=lambda **kw:events.append('stage'))
    runner=N(forward=lambda f:(events.append('draft') or N(logits_output=N(),can_run_graph=True)))
    worker=N(block_size=8,selector=object(),lilicorr=None,_is_domino=False,_draft_sampler=sampler,
             _selector_sample='stale',_draft_block_tokens_buf=torch.empty((bs,8),dtype=torch.int64),
             draft_model_runner=runner,draft_parallel_layout=None,_selector_sampling_enabled=True)
    if proposer:worker._request_lookup_proposer=Fixed8LookupProposer(lambda b:snapshots,enabled=enabled)
    batch=N(sampling_info=N(is_all_greedy=sampling),has_grammar=grammar,return_logprob=False,reqs=[N(rid=x.request_id,_lookup_request_epoch=x.request_epoch) for x in snapshots])
    if anchors is None:anchors=[x.tokens[-1] for x in snapshots]
    if prefixes is None:prefixes=[len(x.tokens)-1 for x in snapshots]
    blocks=torch.zeros((bs,8),dtype=torch.int64);blocks[:,0]=torch.tensor(anchors)
    namespace['get_parallel']=lambda:N(tp_size=world,pp_size=1,attn_dp_enabled=False)
    out=namespace['probe'](worker,batch,blocks,torch.tensor(prefixes),bs,N())
    return out,events,worker

snaps=[Snapshot('a',0,history,64)]
out,events,w=run(snaps)
assert events==[] and out.tolist()==[[4,5,6,7,8,9,10,11]] and w._selector_sample is None
checks.append('all_hit_skips_actual_draft_branch_and_clears_stale_selector')
for kwargs in ({'enabled':False},{'proposer':False},{'world':8},{'grammar':True},{'prefixes':[len(history)-2]},{'anchors':[99]}):
    out,events,w=run(snaps,**kwargs)
    assert events==['stage','draft'] and out[0,1:].tolist()==[777]*7
    checks.append('stock_fallback:'+str(kwargs))
no_match=Snapshot('b',0,tuple(range(60,100)),64)
out,events,w=run(snaps+[no_match])
assert events==['stage','draft'] and out[0,1:].tolist()==[5,6,7,8,9,10,11] and out[1,1:].tolist()==[777]*7
checks.append('mixed_hit_preserves_miss_rows_runs_backbone_once')
# Under output cap<8 this scaffold keeps stock fixed-width path, not a width policy.
out,events,w=run([Snapshot('short',0,history,3)])
assert events==['stage','draft']
checks.append('short_match_or_output_limit_falls_back_no_ragged_width')
# Native source outside this proposal branch must stay exactly the same.
root=P.parents[3];original=(root/'sglang/python/sglang/srt/speculative/dflash_worker_v2.py').read_text()
ao=original.index('        if self.selector is not None or self.lilicorr is not None:\n',original.index('        draft_plan.bind(forward_batch, draft_translator)'))
bo=original.index('        draft_tokens = self._draft_block_tokens_buf[:bs]',ao)
new_end=s.index('        draft_tokens = self._draft_block_tokens_buf[:bs]',a)
assert s[:a]==original[:ao] and s[new_end:]==original[bo:]
checks.append('unchanged_full_cache_prepare_target_verify_aux_append_source')
# Independent review: same shape/anchor is insufficient proof of request provenance.
provider=Fixed8LookupProposer(lambda b:[Snapshot('other',0,history,64)],enabled=True)
bad_batch=N(sampling_info=N(is_all_greedy=True),has_grammar=False,return_logprob=False,
            reqs=[N(rid='actual',_lookup_request_epoch=0)])
w=N(block_size=8,_is_domino=False,lilicorr=None)
try: provider(worker=w,batch=bad_batch,anchor_tokens=torch.tensor([4]),prefix_lens=torch.tensor([len(history)-1]))
except ValueError: pass
else: raise AssertionError('foreign provider snapshot accepted')
provider=Fixed8LookupProposer(lambda b:[Snapshot('actual',1,history,64)],enabled=True)
try: provider(worker=w,batch=bad_batch,anchor_tokens=torch.tensor([4]),prefix_lens=torch.tensor([len(history)-1]))
except ValueError: pass
else: raise AssertionError('stale lifecycle epoch accepted')
checks.append('foreign_snapshot_same_anchor_length_and_stale_lifecycle_rejected')
assert not torch.cuda.is_initialized()
report={'status':'PASS_SOURCE_FRAGMENT_CPU_ONLY','checks':checks,'cuda_initialized':False,'scope':'Actual transformed draft proposal branch with fake model forward/sampler and real CPU tensors. Surrounding cache and target code preserved byte-for-byte, not executed.','held':['No whole-worker model/KV/graph execution','TP8 deliberately bypasses lookup until rank protocol qualified','Host snapshot provider absent from serving; explicit .cpu readback cost unmeasured','Conservative index rebuild on history changes can dominate cost']}
(P/'OVERLAY_CPU_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'cuda_initialized':False}))
