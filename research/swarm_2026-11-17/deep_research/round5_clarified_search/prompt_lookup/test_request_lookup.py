import ast, dataclasses, hashlib, json, random, time
from pathlib import Path
from typing import Tuple
import torch
from request_lookup import RequestLookup

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
assert not torch.cuda.is_initialized(), 'CPU-only probe must not initialize CUDA'

def extract(path, name):
    source=path.read_text(encoding='utf-8')
    tree=ast.parse(source)
    node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
    ns={'torch':torch,'Tuple':Tuple}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[name],hashlib.sha256(source.encode()).hexdigest()

accept,sha_accept=extract(ROOT/'sglang/python/sglang/kernels/ops/speculative/dflash.py','compute_dflash_correct_drafts_and_bonus')
commit,sha_commit=extract(ROOT/'sglang/python/sglang/srt/speculative/dflash_worker_v2.py','_commit_accept')
checks=[]
def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('expected rejection')

# Every first-rejection position including all-match, widths beyond stock8.
for width in (2,4,8,16,32):
    for k in range(width):
        state=RequestLookup('one',list(range(1,50))+[90,91,1,2,3,4],max_rows=width,max_match=4)
        cand=state.propose(remaining_output=64)
        assert cand and len(cand.tokens)==width
        pred=list(cand.tokens[1:])+[12345]
        if k<width-1:pred[k]=54321
        a,b=accept(candidates=torch.tensor([cand.tokens]),target_predict=torch.tensor([pred]))
        outputs,lens=commit(torch.tensor([cand.tokens]),a,b)
        result=state.accept_target_predictions(cand,pred)
        assert result.accepted_drafts==k==a.item()
        assert result.emitted==tuple(outputs[0,:lens.item()].tolist())
        assert result.cached_inputs==cand.tokens[:lens.item()]
        assert result.bonus==b.item()
        assert len(result.emitted)==len(result.cached_inputs)==k+1
        assert state.context[-1]==result.bonus
        reject(lambda:state.accept_target_predictions(cand,pred))
        checks.append({'kind':'native_CPU_accept_contract','width':width,'first_reject':k})

# Width1 fallback native rule; proposals may never be committed without target.
a,b=accept(candidates=torch.tensor([[7]]),target_predict=torch.tensor([[11]]))
out,n=commit(torch.tensor([[7]]),a,b)
assert a.item()==0 and n.item()==1 and out.item()==11

s=RequestLookup('a',[1,2,3,4,5,6,7,8,1,2,3,4],max_match=4)
c=s.propose(remaining_output=3)
assert len(c.tokens)==3
assert s.propose(remaining_output=1) is None
s.validate(c)
reject(lambda:s.validate(dataclasses.replace(c,tokens=(c.tokens[0],999,998))))
reject(lambda:s.validate(dataclasses.replace(c,tokens=(999,)+c.tokens[1:])))
foreign=RequestLookup('b',s.context,max_match=4)
reject(lambda:foreign.validate(c))
s.replace_context(s.context)
reject(lambda:s.validate(c))
s.close()
reject(lambda:s.propose(remaining_output=3))
reject(lambda:RequestLookup('invalid',[True,2,3,4]))
checks.append({'kind':'stale_foreign_tamper_outputcap_closed_and_invalid_token_rejections'})

# EOS is not copied as a draft proposal; target remains free to return EOS bonus.
s=RequestLookup('eos',[1,2,3,4,5,99,7,8,1,2,3,4],max_match=4)
c=s.propose(remaining_output=32,eos_ids=(99,))
assert c.tokens==(4,5)
r=s.accept_target_predictions(c,[5,99])
assert r.emitted==(5,99) and r.cached_inputs==(4,5)
s.close()
checks.append({'kind':'EOS_target_bonus_and_KV_frontier'})

# Causal target oracle: each verifier row predicts under its proposed prefix.
# Compare publication against serial target generation, not against lookup itself.
def target(prefix):
    value=2166136261
    for t in prefix[-17:]:value=((value ^ t)*16777619)&0xffffffff
    return value%29
rng=random.Random(527)
for case in range(100):
    base=[rng.randrange(29) for _ in range(80)]
    history=base+base[20:28]
    s=RequestLookup('causal'+str(case),history,max_rows=(4,8,16,32)[case%4])
    c=s.propose(remaining_output=32)
    assert c
    pred=[target(history[:-1]+list(c.tokens[:i+1])) for i in range(len(c.tokens))]
    result=s.accept_target_predictions(c,pred)
    serial=list(history)
    expected=[]
    for _ in result.emitted:
        token=target(serial);expected.append(token);serial.append(token)
    assert result.emitted==tuple(expected)
    assert s.context==tuple(serial)
checks.append({'kind':'causal_serial_target_equivalence','cases':100,'model':'deterministic CPU token function, not Muse'})

# Long acceptance and repeated versioned commits with known periodic target.
s=RequestLookup('periodic',[i%23 for i in range(230)]+[0,1,2,3],max_rows=32)
for step in range(30):
    c=s.propose(remaining_output=128)
    assert c
    pred=[(t+1)%23 for t in c.tokens]
    r=s.accept_target_predictions(c,pred)
    assert r.emitted==tuple((c.tokens[0]+i+1)%23 for i in range(len(r.emitted)))
checks.append({'kind':'long_acceptance_repeated_history_updates','steps':30})

# Host-only timings establish potential lookup overhead, not an acceptance/gain estimate.
timings=[]
for label,h in [('random',[rng.randrange(202048) for _ in range(74160)]),('periodic',[i%97 for i in range(74160)])]:
    h=h+h[100:116]
    start=time.perf_counter();s=RequestLookup(label,h);build=time.perf_counter()-start
    start=time.perf_counter()
    for _ in range(500):s.propose(remaining_output=220)
    lookup=(time.perf_counter()-start)/500
    timings.append({'synthetic_history':label,'tokens':len(h),'build_ms':build*1000,'lookup_us':lookup*1e6,'index_keys':len(s._index),'retained_endpoints':sum(map(len,s._index.values()))})
assert not torch.cuda.is_initialized()
report={'status':'PASS_CPU_CONTRACT_ONLY','checks':checks,'timings':timings,'source_sha256':{'dflash.py':sha_accept,'dflash_worker_v2.py':sha_commit},'torch':torch.__version__,'cuda_initialized':False,'limits':['No actual Muse target, model weights, GPU graph, target/draft KV or serving timing.','Native CPU branch and commit function extracted unchanged; CUDA accept kernel not executed.','Synthetic lookup timings do not predict real match rates or target acceptance.','Not installed in DFlash worker; synchronous/overlap host frontier integration still held.']}
(HERE/'CPU_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'native_rejection_positions':sum(x['kind']=='native_CPU_accept_contract' for x in checks),'timings':timings,'cuda_initialized':False}))
