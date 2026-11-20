"""Produce separate default-off graph-aware overlay; preserve eager artifact."""
import ast,difflib,hashlib,json
from pathlib import Path
import build_eager_prefix_overlay as eager
P=Path(__file__).parent;ROOT=P.parents[3];OUT=P/'graph_prefix_overlay'

def change_worker(source):
    text=eager.worker_change(source)
    text=eager.once(text,'import torch\n','import torch\n\nfrom sglang.srt.speculative.target_prefix_graph_runtime import (\n    initialize_target_prefix_states, target_prefix_scope,\n)\n')
    text=eager.once(text,'    def init_cuda_graphs(self):\n','    def init_cuda_graphs(self):\n        # Default-off; target capture exists, draft TP scope has not started.\n        initialize_target_prefix_states(self)\n')
    def scope(fn):
        marker='        # --- 2) Target verify.\n'
        before,tail=fn.split(marker)
        # All early returns/exceptions restore the target state. Full draft8
        # proposal execution above this point is outside the scope.
        return before+'        with target_prefix_scope(self, verify_block_size):\n'+''.join('    '+line if line.strip() else line for line in (marker+tail).splitlines(True))
    return eager.replace_function(text,'DFlashWorkerV2','forward_batch_generation',scope)

def main():
    changes={
      'python/sglang/srt/speculative/dflash_worker_v2.py':change_worker,
      'python/sglang/srt/layers/attention/flashattention_backend.py':eager.fa3_change}
    outputs={};sources={};diff=[]
    for rel,fn in changes.items():
        original=(ROOT/'sglang'/rel).read_text();outputs[rel]=fn(original)
        sources[rel]=hashlib.sha256(original.encode()).hexdigest()
        diff.extend(difflib.unified_diff(original.splitlines(True),outputs[rel].splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    for name in ('dflash_prefix_verification.py','target_verify_runtime_state.py','target_prefix_graph_runtime.py'):
        rel='python/sglang/srt/speculative/'+name
        text=(P/name).read_text(encoding='utf-8-sig').replace('from target_verify_runtime_state import','from sglang.srt.speculative.target_verify_runtime_state import')
        outputs[rel]=text
        diff.extend(difflib.unified_diff([],text.splitlines(True),fromfile='/dev/null',tofile='b/'+rel))
    for rel,text in outputs.items():
        compile(text,rel,'exec');path=OUT/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    patch=''.join(diff);(P/'graph_prefix_overlay.diff').write_text(patch)
    report={'status':'DEFAULT_OFF_SOURCE_SCAFFOLD_HELD','active_checkout_changed':False,
      'activation':'RESEARCH_TARGET_VERIFY_WIDTHS=() default-off. Explicit isolated fixture enable builds target states; configure_prefix_verification selects fixedW.',
      'real_factory':'Pinned _get_attention_backend(init_new_workspace=True) + DecodeCudaGraphRunner explicit attn_backend/width; model/KV/globaldraft unchanged',
      'scope':'Post-proposal target preparation, model forward, accept, publication, auxiliary consumption; finally restores default target backend/graph',
      'startup':'DFlash.init_cuda_graphs entry, after scheduler target capture, before draft TP context and before post-capture KV resize',
      'holds':['Actual target model/attention graph startup and numeric greedy equivalence across widths',
        'Before all weight mutations and recaptures, caller MUST invoke invalidate_target_prefix_states; generic production hooks not installed',
        'All TP/EP ranks must enable/select same widths; no controller/rank handshake implemented',
        'Additional graph workspace/cubin/peak sizing, not just logical metadata bytes',
        'Other capture-tail extensions deliberately rejected; argmax overlay not composable without separate integration',
        'No serving or performance qualification'],
      'sources':sources,'outputs':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in outputs.items()},
      'diff_sha256':hashlib.sha256(patch.encode()).hexdigest(),'diff_bytes':len(patch.encode()),
      'added_lines':sum(l.startswith('+') and not l.startswith('+++') for l in patch.splitlines()),
      'removed_lines':sum(l.startswith('-') and not l.startswith('---') for l in patch.splitlines())}
    (P/'GRAPH_PREFIX_OVERLAY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()