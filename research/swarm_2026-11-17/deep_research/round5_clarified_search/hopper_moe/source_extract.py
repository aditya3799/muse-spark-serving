"""Extract pinned methods/call expressions without importing CUDA/SGLang."""
import ast,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'RULES.md').exists())
FA=ROOT/'sglang/python/sglang/srt/layers/attention/flashattention_backend.py'
BASE=ROOT/'sglang/python/sglang/srt/layers/attention/base_attn_backend.py'
GRAPH=ROOT/'sglang/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py'
def method(path,cls,name,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.decorator_list=[]
    text=ast.unparse(fn)
    exec(compile('from __future__ import annotations\n'+text,str(path),'exec'),namespace)
    return namespace[name],hashlib.sha256(text.encode()).hexdigest()
def attention_call(phase):
    tree=ast.parse(FA.read_text(encoding='utf-8'))
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashAttentionBackend')
    fn=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==('forward_decode' if phase=='decode' else 'forward_extend'))
    calls=[]
    for call in ast.walk(fn):
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='flash_attn_with_kvcache':continue
        kw={k.arg:k.value for k in call.keywords}
        if 'out' not in kw or not isinstance(kw['out'],ast.Name) or kw['out'].id!='_fa_out':continue
        split=kw.get('num_splits')
        if phase=='decode' and isinstance(split,ast.IfExp):calls.append(call)
        if phase=='verify' and isinstance(split,ast.Attribute) and split.attr=='num_splits' and isinstance(kw.get('page_table'),ast.Name) and kw['page_table'].id=='page_table' and 'max_seqlen_q' in kw:
            q=kw.get('q')
            if isinstance(q,ast.Call) and isinstance(q.func,ast.Attribute) and q.func.attr=='view':calls.append(call)
    if len(calls)!=1:raise RuntimeError(('source callsite changed',phase,len(calls)))
    text=ast.unparse(calls[0]);return compile(text,str(FA),'eval'),text

def make_forward(phase,flash_fn):
    call,text=attention_call(phase)
    def forward(self,q,k,v,layer,forward_batch,**unused):
        md=self.forward_metadata
        env=dict(self=self,q=q,q_reshaped=q.contiguous().view(-1,layer.tp_q_head_num,layer.head_dim),layer=layer,
          key_cache=self.key_cache,value_cache=self.value_cache,page_table=md.page_table,cache_seqlens=md.cache_seqlens_int32,cu_seqlens_q=md.cu_seqlens_q,cu_seqlens_k=getattr(md,'cu_seqlens_k',None),max_seqlen_q=md.max_seq_len_q,metadata=md,
          use_local_attn=False,use_cascade_attn=False,causal=True,window_size=getattr(layer,'window_size',(-1,-1)),return_lse=False,
          is_swa_layer=getattr(layer,'is_swa',False),pa_swa_active=False,sched_meta=None if getattr(layer,'is_swa',False) else getattr(md,'scheduler_metadata',None),_fa_out=None,
          kwargs=getattr(self,'native_kwargs',{}),flash_attn_with_kvcache=flash_fn)
        return eval(call,env)
    return forward,hashlib.sha256(text.encode()).hexdigest()
