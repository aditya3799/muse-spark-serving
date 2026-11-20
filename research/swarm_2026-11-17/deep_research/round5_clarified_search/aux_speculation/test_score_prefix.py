"""Source-extracted CPU gates; synthetic scores, no calibration or GPU evidence."""
import ast,copy,hashlib,json,math,threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as N
import torch
from torch import nn
from score_prefix_controller import *
from source_extract import selector_methods,extract,WORKER,MODEL
from build_score_overlay import patched_source
P=Path(__file__).parent
torch.set_num_threads(1)
torch.set_grad_enabled(False)
counts={}; negatives=[]

def check(name,condition):
    assert condition,name
    counts[name]=counts.get(name,0)+1

def rejects(name,fn):
    try: fn()
    except (RuntimeError,ValueError,TypeError): negatives.append(name)
    else: raise AssertionError('missing rejection: '+name)

def selector_fixture(seed=0,k=16):
    torch.manual_seed(seed)
    env=selector_methods()
    selector=N(top_k=k,predecessor_codebook=torch.randn(128,8),
               successor_codebook=torch.randn(128,8),hidden_projection=nn.Linear(32,8,bias=False))
    selector.build_lattice=lambda **kw:env['build_lattice'](selector,**kw)
    selector.sample_path=lambda **kw:env['sample_path'](selector,**kw)
    ids=torch.stack([torch.randperm(128)[:k] for _ in range(7)])[None]
    unary=torch.randn(1,7,k); hidden=torch.randn(1,7,32);anchor=torch.tensor([77])
    score=selector.build_lattice(candidate_ids=ids,unary_logits=unary,hidden_states=hidden,anchor_token_ids=anchor)
    tokens,q=selector.sample_path(candidate_ids=ids,scores=score,uniforms=torch.rand(1,7),
        temperatures=torch.ones(1),greedy_mask=torch.ones(1,dtype=torch.bool))
    return selector,ids,unary,hidden,anchor,score,tokens,q


def scalar_oracle(ids,scores,tokens):
    result=[];prev=0
    for j in range(7):
        match=(ids[0,j]==tokens[0,j]).nonzero().flatten()
        assert len(match)==1
        selected=int(match[0]);row=scores[0,j,prev]
        assert int(row.argmax())==selected
        other=torch.cat((row[:selected],row[selected+1:])).max()
        result.append(float(row[selected]-other));prev=selected
    return result


def run_numeric():
    for seed in range(40):
        selector,ids,unary,hidden,anchor,scores,tokens,q=selector_fixture(seed)
        out=torch.empty(1,8);export_path_margins(ids,scores,tokens,out)
        check('actual_lattice_walk_margin',out[0,:7].tolist()==scalar_oracle(ids,scores,tokens))
        check('actual_walk_valid',out[0,7].item()==1)
        check('greedy_q_is_one_hot',bool(((q==0)|(q==1)).all() and (q.sum(-1)==1).all()))
        # Caller-owned snapshot survives producer temporary overwrite.
        saved=out.clone();scores.fill_(float('nan'))
        check('independent_export',torch.equal(out,saved))
    # Same greedy path and q, arbitrarily different score confidence.
    ids=torch.arange(16).repeat(1,7,1);tokens=torch.zeros(1,7,dtype=torch.int64)
    low=torch.zeros(1,7,16,16);low[:,:,:,0]=.01
    high=low*1000;outputs=[]
    for scores in (low,high):
        out=torch.empty(1,8);export_path_margins(ids,scores,tokens,out);outputs.append(out)
    policy=Policy(True,1.)
    snaps=[Snapshot(0,1,'request',tuple(o[0,:7].tolist()),bool(o[0,7])) for o in outputs]
    check('same_proposals_different_width', [policy.choose(s) for s in snaps]==[4,8])
    # Wrong predecessor-row shortcut demonstrably differs from actual walk.
    selector,ids,_,_,_,scores,tokens,_=selector_fixture(3)
    correct=scalar_oracle(ids,scores,tokens)
    wrong=[]
    for j in range(7):
        idx=int((ids[0,j]==tokens[0,j]).nonzero()[0]); row=scores[0,j,0]
        wrong.append(float(row[idx]-torch.cat((row[:idx],row[idx+1:])).max()))
    check('wrong_predecessor_falsified',correct!=wrong)
    for mutation in ('missing_token','duplicate_selected','nan_row','inf_row','nongreedy_token'):
        _,ids,_,_,_,scores,tokens,_=selector_fixture(2)
        ids=ids.clone();scores=scores.clone();tokens=tokens.clone()
        if mutation=='missing_token':tokens[0,0]=999
        if mutation=='duplicate_selected':ids[0,0,:]=tokens[0,0]
        if mutation=='nan_row':scores[0,0,0,0]=float('nan')
        if mutation=='inf_row':scores[0,0,0,0]=float('inf')
        if mutation=='nongreedy_token':tokens[0,0]=ids[0,0,(int((ids[0,0]==tokens[0,0]).nonzero()[0])+1)%16]
        out=torch.empty(1,8);export_path_margins(ids,scores,tokens,out)
        check('invalid_path_'+mutation,out[0,7].item()==0)
        check('invalid_fallback_'+mutation,policy.choose(Snapshot(0,1,'r',tuple(out[0,:7].tolist()),False))==8)
    rejects('output_alias',lambda:export_path_margins(ids,scores,tokens,scores.flatten()[:8].view(1,8)))
    rejects('wrong_dtype',lambda:export_path_margins(ids,scores.half(),tokens,torch.empty(1,8)))
    rejects('wrong_width',lambda:export_path_margins(ids[:,:6],scores[:,:6],tokens[:,:6],torch.empty(1,8)))
    rejects('uncalibrated_enabled_missing_threshold',lambda:Policy(True).choose(snaps[0]))
    check('disabled_identity',Policy().choose(snaps[0])==8)
    for field,value in [('margins',(1.,)*6),('margins',(float('nan'),)*7),('valid',1),('generation',0),('request_key','')]:
        rejects('malformed_'+field,lambda:policy.choose(replace(snaps[0],**{field:value})))
    check('threshold_boundary',Policy(True,.01).choose(replace(snaps[0],margins=(.01,)*7))==8)


def run_owner():
    _,ids,_,_,_,scores,tokens,_=selector_fixture(5)
    owner=ScoreExport();graph=object();owner.begin_capture();owner.capture_write(ids,scores,tokens)
    rejects('capture_unpublished',lambda:owner.consume_blocking(0,3,'r'))
    owner.finish_capture(graph,3)
    for i in range(4):
        gen=owner.begin_replay(graph,3,'r'+str(i));owner.capture_write(ids,scores,tokens)
        rejects('premature_consume',lambda:owner.consume_blocking(gen,3,'r'+str(i)))
        rejects('duplicate_launch',lambda:owner.begin_replay(graph,3,'another'))
        rejects('stale_publish',lambda:owner.finish_replay_enqueue(gen-1))
        owner.finish_replay_enqueue(gen)
        rejects('wrong_request',lambda:owner.consume_blocking(gen,3,'other'))
        rejects('wrong_stream',lambda:owner.consume_blocking(gen,4,'r'+str(i)))
        snap=owner.consume_blocking(gen,3,'r'+str(i));old=snap.margins
        owner.out.fill_(0)
        check('host_snapshot_independent',snap.margins==old)
        rejects('duplicate_consume',lambda:owner.consume_blocking(gen,3,'r'+str(i)))
    rejects('replaced_graph',lambda:owner.begin_replay(object(),3,'r'))
    rejects('rebind_data',lambda:(setattr(owner.out,'data',torch.empty_like(owner.out)),owner.guard()))
    owner=ScoreExport();owner.begin_capture();owner.abort('capture failed')
    rejects('failed_capture_held',lambda:owner.finish_capture(object(),3))
    owner=ScoreExport();owner.begin_capture();owner.capture_write(ids,scores,tokens);owner.finish_capture(graph,3)
    gen=owner.begin_replay(graph,3,'r');owner.abort('producer error')
    rejects('producer_error_not_recycled',lambda:owner.begin_replay(graph,3,'r'))
    owner=ScoreExport();owner.begin_capture()
    rejects('missing_export_capture',lambda:owner.finish_capture(graph,3))
    owner=ScoreExport();owner.out.resize_(8,1)
    rejects('resize_binding',owner.guard)
    owner=ScoreExport();errs=[]
    def foreign():
        try: owner.guard()
        except RuntimeError:errs.append(True)
    t=threading.Thread(target=foreign);t.start();t.join()
    check('foreign_thread',errs==[True])


def run_ranks():
    snap=Snapshot(2,9,'own-request',(.2,)*7,True,tuple(range(7)));policy=Policy(True,1.)
    base=make_packet(snap,policy,tuple(range(7)))
    rejects('captured_proposal_mismatch',lambda:make_packet(snap,policy,(9,)*7))
    for rank in range(8):
        packets=[base]*8;packets[rank]=replace(base,width=8)
        check('canonical_rank_width',agree_packets(packets,8)==packets[0].width)
    for field,value in [('epoch',3),('generation',10),('request_key','other'),
                        ('policy_signature',('other',)),('proposed_tokens',(9,)*7),('width',6)]:
        packets=[base]*8;packets[3]=replace(base,**{field:value})
        rejects('rank_mismatch_'+field,lambda:agree_packets(packets,8))
    rejects('missing_rank',lambda:agree_packets([base]*7,8))


def run_sampler_source():
    env=selector_methods();extract(WORKER,['_selector_lattice'],namespace=env)
    _,patched=patched_source();tree=ast.parse(patched)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_SelectorDraftSampler')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(WORKER),'exec'),env)
    sel,ids,unary,hidden,anchor,_,_,_=selector_fixture(14)
    model=N(candidate_selector=sel,compute_candidates=lambda h:(ids.flatten(0,1),unary.flatten(0,1)))
    sampler=env['_SelectorDraftSampler'](draft_model=model,block_size=8,max_bs=1,device='cpu',sampling_enabled=False)
    hidden8=torch.cat((torch.zeros(1,1,32),hidden),1).flatten(0,1)
    inputids=torch.cat((anchor[:,None],torch.zeros(1,7,dtype=torch.int64)),1).flatten()
    check('source_hook_default_off',sampler._research_score_export is None)
    sampler(hidden8,inputids);baseline=sampler.out.clone();q=sampler.q_out.clone()
    owner=ScoreExport();sampler._research_score_export=owner;owner.begin_capture()
    sampler(hidden8,inputids);owner.finish_capture(sampler,4)
    gen=owner.begin_replay(sampler,4,'source-request');sampler(hidden8,inputids)
    owner.finish_replay_enqueue(gen);snap=owner.consume_blocking(gen,4,'source-request')
    check('source_hook_preserves_proposals',torch.equal(sampler.out,baseline))
    check('source_hook_preserves_q',torch.equal(sampler.q_out,q))
    check('source_hook_valid',snap.valid)
    check('source_hook_captured_tokens',snap.proposed_tokens==tuple(baseline.tolist()))
    check('source_hook_margin',snap.margins==tuple(scalar_oracle(ids,sel.build_lattice(candidate_ids=ids,unary_logits=unary,hidden_states=hidden,anchor_token_ids=anchor),baseline.view(1,7))))

if __name__=='__main__':
    run_numeric();run_owner();run_ranks();run_sampler_source()
    result={'status':'CPU_SOURCE_COMPONENT_PASS','checks':counts,'check_total':sum(counts.values()),
      'negative_controls':negatives,'negative_count':len(negatives),
      'scope':'Actual extracted lattice, greedy walk and patched sampler methods on CPU. Top-k/LM projection mocked. Synthetic scores and thresholds, no calibration. CPU lifecycle/rank model, no real graph, CUDA, TP or serving validation.',
      'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MODEL,WORKER,P/'score_prefix_controller.py',P/'source_extract.py',P/'test_score_prefix.py',P/'build_score_overlay.py']}}
    (P/'CPU_SCORE_PREFIX_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'checks':result['check_total'],'negative_count':len(negatives)}))



