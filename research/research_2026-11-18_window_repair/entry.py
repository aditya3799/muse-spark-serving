"""Exercise the patched DFlash worker and stock RadixKey with CPU operator fixtures."""
import ast
from array import array
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace as S
import subprocess
from typing import Optional
import torch
from namespaces import DFlashWindowNamespaces
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
WORK="python/sglang/srt/speculative/dflash_worker_v2.py"
def extract(source,owner,name,ns):
    parsed=ast.parse(source)
    cls=next(n for n in parsed.body if getattr(n,"name",None)==owner)
    node=next(n for n in cls.body if getattr(n,"name",None)==name)
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),name,"exec"),ns)
    return ns[name]
class Mode:
    def is_extend(self):return True
    def is_idle(self):return False
MODE=Mode()
def stock(name):
    return subprocess.check_output(["git","-C",str(ROOT/"sglang"),"show",PIN+":"+name]).decode()

def req(ids,salt=None,session=None):
    return S(origin_input_ids=ids,cache_salt=salt,extra_key=None,session=session,multimodal_inputs=None,
             positional_embed_overrides=None,skip_radix_cache_insert=False,output_ids=[])
def route_test():
    sol=(HERE/"applied"/WORK).read_text()
    def position(_,pref,ext,m):
        return torch.cat([torch.arange(int(a),int(a+b)) for a,b in zip(pref,ext)]),None
    ns=dict(ScheduleBatch=object,GenerationBatchResult=object,CaptureHiddenMode=S(FULL=True),
            SpecTpSyncSite=S(DFLASH_TARGET=True),ForwardMode=S(EXTEND=MODE),
            async_h2d=lambda x,**kw:torch.tensor(x,**kw),compute_position=position,torch=torch)
    methods=[extract(v,"DFlashWorkerV2","forward_batch_generation",ns.copy()) for v in (stock(WORK),sol)]
    calls=[[],[]]
    def form(i):
        def target(batch,**kwargs):
            n=sum(batch.extend_lens)
            return S(kv_loc_plan=torch.arange(n),logits_output=S(hidden_states=torch.arange(n)[:,None]),
                     next_token_ids=torch.tensor([17]*len(batch.reqs)))
        return S(_validate_phase1_sampling_support=lambda *a:None,_wait_for_fused_hicache_load=lambda *a:None,
                 target_worker=S(forward_batch_generation=target),_tp_sync=S(sync=lambda *a:None),
                 model_runner=S(prefill_attention_backend_str="fa3"),_draft_write_ids=lambda plan:plan,
                 _make_next_draft_input_prefill=lambda **kw:kw,
                 _append_target_hidden_to_draft_kv_by_loc=lambda **kw:calls[i].append(kw["positions"].tolist()))
    workers=[form(0),form(1)]
    workers[1]._window_namespaces=space=DFlashWindowNamespaces(8,4)
    a=req(list(range(80))); short=req(list(range(15)))
    space.assign(a);space.assign(short)
    checks=[]
    for requests,prefix,ext in (([a],[0],[16]),([a],[48],[16]),([a,short],[0,0],[16,15]),([a],[64],[16])):
        batch=S(forward_mode=MODE,is_extend_in_batch=False,reqs=requests,prefix_lens=prefix,extend_lens=ext,
                seq_lens=torch.tensor([b+c for b,c in zip(prefix,ext)]),out_cache_loc=torch.arange(sum(ext)))
        before=len(calls[1])
        outputs=[fn(w,batch) for fn,w in zip(methods,workers)]
        assert torch.equal(outputs[0].next_token_ids,outputs[1].next_token_ids)
        assert torch.equal(outputs[0].next_draft_input["seq_lens"],outputs[1].next_draft_input["seq_lens"])
        assert all(o.logits_output.hidden_states is None for o in outputs)
        checks.append(len(calls[1])==before)
    assert checks==[True,False,False,False],checks
    return checks

def branch_test():
    radix=stock("python/sglang/srt/mem_cache/radix_cache.py")
    child=extract(radix,"RadixKey","child_key_at",{})
    class Key:
        def __init__(self,ids,salt):
            self.token_ids=ids;self.cache_salt=salt;self.extra_key=None;self.is_bigram=False
        def __len__(self):return len(self.token_ids)
    from collections import Counter
    cnt=Counter()
    for sd in range(50):
        rnd=random.Random(sd)
        # Deliberately tiny hash tables induce hint collisions.
        space=DFlashWindowNamespaces(8,4,1+sd%4)
        cache={}
        corpus=[list(range(94))]
        for turn in range(50):
            base=rnd.choice(corpus)
            prompt=(base[:rnd.randint(1,len(base))]+[rnd.randint(100,130) for _ in range(rnd.randint(1,45))]) if turn else base
            r=req(prompt,salt=("customer" if (turn%9==0) else None))
            space.assign(r); sl=r.cache_salt;space.assign(r);assert r.cache_salt==sl
            key=Key(prompt,sl); root=child(key,0,1)[0]
            # Pinned KV keys include this root at every split.
            nodes=cache.setdefault(root,{})
            positions=[]
            while len(positions)<len(prompt)-1:
                i=len(positions);v=nodes.get(tuple(prompt[:i+1]))
                if v is None:break
                positions.append(v)
            positions=positions[:len(positions)//4*4]
            while len(positions)<len(prompt):
                p=len(positions);e=min(16,len(prompt)-p)
                skip=space.can_omit([r],[p],[e])
                cnt["skipped_rows"]+=e*skip
                positions.extend([not skip]*e)
                for i in range(len(positions)//4*4):
                    positions[i]=nodes.setdefault(tuple(prompt[:i+1]),positions[i])
                if turn%11==0:
                    cache[root]=nodes=dict(nodes)
            for _ in range(rnd.randint(1,6)):
                assert all(positions[-8:]);cnt["checked_reads"]+=min(8,len(positions))
                positions.append(True);prompt=prompt+[rnd.randrange(100,130)]
            for i in range(len(positions)//4*4):
                positions[i]=nodes.setdefault(tuple(prompt[:i+1]),positions[i])
            assert all(positions[-8:])
            corpus.append(prompt)
            if turn%17==0:cache.clear()
            cnt["requests"]+=1
    return dict(cnt)
if __name__=="__main__":
    d=dict(eligible_worker_omit=route_test(),physical_tree=branch_test(),
           source="CPU stock/patched forward_batch_generation with target, operator and transfer fixtures; source-extracted RadixKey.child_key_at",
           sha256=hashlib.sha256((HERE/"candidate.diff").read_bytes()).hexdigest())
    (HERE/"entry.json").write_text(json.dumps(d,indent=2))
    print(d)

