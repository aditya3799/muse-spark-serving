"""Exact integer multilayer causal model for fixed8 expert-tail masking.
This tests the algorithm/lifetimes, not floating-point GPU implementation.
"""
import hashlib,json,random
from pathlib import Path
from prefix_protocol import accept_prefix
P=Path(__file__).resolve().parent
VOCAB=23;MOD=65521;LAYERS=4

def forward(tokens,cache,verify_width,window):
    # Preserve full token shape through all layers. Only expert-tail work is absent.
    hidden=[(token*17+13)%MOD for token in tokens]
    writes=[];aux=[];expert_rows=0
    for layer in range(LAYERS):
        k=[(h*3+layer)%MOD for h in hidden]
        v=[(h*7+layer+1)%MOD for h in hidden]
        entries=list(zip(k,v));base=len(cache[layer]);combined=cache[layer]+entries
        result=[]
        for row,h in enumerate(hidden):
            pos=base+row;start=max(0,pos-window+1) if window else 0
            attention=sum(((h*kk+5)%97)*vv for kk,vv in combined[start:pos+1])%MOD
            residual=(h+attention)%MOD
            if row<verify_width:
                # Sparse row-local experts, two routed selections.
                expert=(residual%11)*residual+((residual//11)%13)*(residual+19)
                expert_rows+=1
            else:expert=0
            result.append((residual+expert)%MOD)
        hidden=result;writes.append(entries);aux.append(list(hidden))
    return [h%VOCAB for h in hidden],writes,aux,expert_rows

def append(cache,writes,count):
    for layer in range(LAYERS):cache[layer].extend(writes[layer][:count])

def prefill(prompt,window):
    cache=[[] for _ in range(LAYERS)]
    for token in prompt:
        _,writes,_,_=forward([token],cache,1,window);append(cache,writes,1)
    return cache

def greedy(prompt,count,window):
    cache=prefill(prompt[:-1],window);anchor=prompt[-1];out=[]
    for _ in range(count):
        predictions,writes,_,_=forward([anchor],cache,1,window)
        append(cache,writes,1);anchor=predictions[0];out.append(anchor)
    return out

rng=random.Random(64032);checks=0;cycles=0;masked_expert_rows=0;negative=0
for window in (0,8,16):
 for width in (1,2,4,6,8):
  for case in range(8):
    prompt=[rng.randrange(VOCAB) for _ in range(1+case)]
    cache=prefill(prompt[:-1],window);anchor=prompt[-1];visible=[];history=list(prompt)
    count=1+case*5;eos=(case*3)%VOCAB if case%2 else None
    reference=greedy(prompt,count,window)
    if eos is not None and eos in reference:reference=reference[:reference.index(eos)+1]
    while len(visible)<count:
        # One full8 draft is built before any target prefix decision.
        predicted=greedy(history,7,window)
        first_wrong=(case+cycles)%8
        proposals=[x if i<first_wrong else (x+1+i)%VOCAB for i,x in enumerate(predicted)]
        candidates=[anchor]+proposals
        full,fullkv,fullaux,work=forward(candidates,cache,8,window)
        masked,maskedkv,maskedaux,maskedwork=forward(candidates,cache,width,window)
        assert full[:width]==masked[:width]
        assert all(a[:width]==b[:width] for a,b in zip(fullkv,maskedkv))
        assert all(a[:width]==b[:width] for a,b in zip(fullaux,maskedaux))
        # Wrongly masking the FIRST real expert row must be detectable somewhere.
        if width==1 and case>0:
            wrong,_,_,_=forward(candidates,cache,0,window)
            if wrong[0]!=full[0]:negative+=1
        commit=accept_prefix(candidates[:width],masked[:width],len(cache[0]))
        append(cache,maskedkv,commit.committed)
        emitted=list(commit.emitted)
        visible_new=emitted[:count-len(visible)]
        stop=eos is not None and eos in visible_new
        if stop:visible_new=visible_new[:visible_new.index(eos)+1]
        visible.extend(visible_new)
        # Committed KV is for inputanchor + accepted proposals; bonus pending.
        history.extend(emitted);anchor=commit.bonus
        assert len(cache[0])==len(history)-1
        for layer in range(LAYERS):assert len(cache[layer])==len(cache[0])
        # Recompute committed prefix independently to catch tail pollution.
        assert cache==prefill(history[:-1],window)
        masked_expert_rows+=work-maskedwork;cycles+=1;checks+=1
        if stop:break
    assert visible==reference;checks+=1
assert negative>0 and masked_expert_rows>0
report={'status':'PASS','request_cases':120,'cycle_checks':cycles,'total_checks':checks,'expert_row_layer_evaluations_skipped':masked_expert_rows,
        'invalid_first_row_mask_detected_cases':negative,'scope':'Exact integer four-layer causal full/SWA model; full8 shape retained, expert suffix masked, real-prefix logits/all-layerKV/aux exact and greedy stream/EOS/outputbudget preserved. Not floating GPU/model proof.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(P/'cpu_full8_tailmask_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
