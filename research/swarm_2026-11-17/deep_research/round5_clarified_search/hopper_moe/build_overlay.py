"""Emit isolated default-off overlay against hash-pinned active source files."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=next(p for p in P.parents if (p/'RULES.md').exists());OUT=P/'overlay'
SUPPORT='sglang.srt.layers.attention.fa3_split_policy'
def edit(source,cls,decorators,tail=None):
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    insert={}
    for fn in node.body:
        if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
        if fn.name in decorators:insert[fn.lineno-1]='    @'+decorators[fn.name]+'\n'
        if tail and fn.name==tail[0]:insert[fn.end_lineno]='        '+tail[1]+'\n'
    if len(insert)!=len(decorators)+(1 if tail else 0):raise RuntimeError('source methods changed')
    lines=source.splitlines(True)
    for pos,text in sorted(insert.items(),reverse=True):lines.insert(pos,text)
    text=''.join(lines)
    marker='from __future__ import annotations\n'
    imports=sorted(set(x.split('(')[0] for x in decorators.values())|({'initialize_target_fa3_split_policy'} if tail else set()))
    insertion='from '+SUPPORT+' import ('+', '.join(imports)+')\n'
    if marker in text:text=text.replace(marker,marker+'\n'+insertion,1)
    else:
        # Place after module docstring, before first import.
        t=ast.parse(text);first=next(n for n in t.body if isinstance(n,(ast.Import,ast.ImportFrom)));ls=text.splitlines(True);ls.insert(first.lineno-1,insertion);text=''.join(ls)
    if cls=='FlashAttentionBackend':
        copy='self._sched_meta_buf[:n] = sched\n                        self._sched_meta_buf[n:] = 0'
        if text.count(copy)!=2:raise RuntimeError('scheduler copy source changed')
        text=text.replace(copy,copy+'\n                        record_scheduler_copy(self, sched, self._sched_meta_buf, n)')
        text=text.replace('from '+SUPPORT+' import (','from '+SUPPORT+' import (record_scheduler_copy, ',1)
    compile(text,cls,'exec');return text
jobs={
 'python/sglang/srt/layers/attention/flashattention_backend.py':('FlashAttentionBackend',{'init_forward_metadata':'fa3_split_scope("metadata")','init_forward_metadata_out_graph':'fa3_split_scope("metadata")','init_forward_metadata_in_graph':'fa3_split_scope("metadata")','forward_extend':'fa3_split_scope("forward")','forward_decode':'fa3_split_scope("forward")','_compute_scheduler_metadata':'fa3_scheduler_guard'},('__init__','initialize_target_fa3_split_policy(self, model_runner)')),
 'python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py':('DecodeCudaGraphRunner',{'capture_one_shape':'fa3_runner_capture_guard'},None),
 'python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py':('FullCudaGraphBackend',{'capture_one':'fa3_graph_capture_guard','replay':'fa3_graph_replay_guard','can_run':'fa3_graph_can_run_guard','cleanup':'fa3_graph_cleanup_guard'},None)}
manifest={'status':'DEFAULT_OFF_ISOLATED_SOURCE_OVERLAY_NOT_INSTALLED','sources':{},'outputs':{},'holds':['native H200 binary precision audit','actual model-runner admission and all-rank agreement','native graph/multirank execution','metadata glue/TBO/PDMux/ragged not admitted','failed capture poisons controller; rebuild backend and graphs']};diff=[]
for rel,(cls,decs,tail) in jobs.items():
 source=(R/'sglang'/rel).read_text(encoding='utf-8');result=edit(source,cls,decs,tail);path=OUT/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(result,encoding='utf-8')
 manifest['sources'][rel]=hashlib.sha256(source.encode()).hexdigest();manifest['outputs'][rel]=hashlib.sha256(result.encode()).hexdigest();diff+=list(difflib.unified_diff(source.splitlines(True),result.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
rel='python/sglang/srt/layers/attention/fa3_split_policy.py';source=(P/'fa3_split_policy.py').read_text(encoding='utf-8-sig');compile(source,rel,'exec');path=OUT/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source,encoding='utf-8');manifest['outputs'][rel]=hashlib.sha256(source.encode()).hexdigest();diff+=list(difflib.unified_diff([],source.splitlines(True),fromfile='/dev/null',tofile='b/'+rel));patch=''.join(diff);(P/'fa3_split_policy.diff').write_text(patch,encoding='utf-8');manifest['patch_sha256']=hashlib.sha256(patch.encode()).hexdigest();manifest['patch_bytes']=len(patch.encode());(P/'OVERLAY_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(manifest['status'],manifest['patch_bytes'])
