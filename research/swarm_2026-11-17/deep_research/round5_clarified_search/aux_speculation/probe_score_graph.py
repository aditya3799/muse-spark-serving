"""Bounded B1 actual selector Triton walk + score-owner graph component gate.
No target, backbone or LM head. Native scores derive from reduced32->8 fixture.
"""
import faulthandler
faulthandler.enable();faulthandler.dump_traceback_later(20,repeat=True)
print('PHASE before imports',flush=True)
import argparse,ast,hashlib,importlib.util,json,time
from pathlib import Path
from types import SimpleNamespace as N
import torch
print('PHASE imported torch',flush=True)
from torch import nn
from score_prefix_controller import ScoreExport,Policy,make_packet
from source_extract import MODEL,WORKER,ROOT,selector_methods,extract
from build_score_overlay import patched_source
P=Path(__file__).parent
pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);args=pa.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
report={'status':'STARTED','checks':0,'negatives':[],'memory':[],
        'scope':'Actual extracted lattice/sampler plus pinned native selector_walk_triton; reduced synthetic hidden/codebooks and mocked upstream candidate projection. B1 only, no full model/target/TP or score calibration.'}

def save(): (args.output/'results.json').write_text(json.dumps(report,indent=2))
def check(ok,name):
    if not ok:raise AssertionError(name)
    report['checks']+=1

def reject(name,fn):
    try:fn()
    except (ValueError,RuntimeError):report['negatives'].append(name)
    else:raise AssertionError('missing rejection '+name)

def main():
    print('PHASE main pre CUDA',flush=True)
    torch.set_grad_enabled(False);torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    print('PHASE before CUDA init',flush=True)
    torch.cuda.init();print('PHASE CUDA initialized',flush=True);entry=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    def reserve(phase):
        free,total=torch.cuda.mem_get_info();peak=torch.cuda.max_memory_allocated()-entry
        report['memory'].append({'phase':phase,'free':free,'peak_own_allocated':peak,
                                 'own_reserved':torch.cuda.memory_reserved()})
        if free<512*2**20 or peak>64*2**20:raise RuntimeError('memory reserve/budget failed')
    free,_=torch.cuda.mem_get_info()
    if free<(512+64)*2**20:
        report['status']='SKIPPED_MEMORY_RESERVE';save();return
    reserve('before_fixture')
    kernels=ROOT/'sglang/python/sglang/kernels/ops/speculative/dflash.py'
    spec=importlib.util.spec_from_file_location('pinned_dflash_kernels',kernels)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    print('PHASE kernels imported',flush=True)
    env=selector_methods();env['selector_walk_triton']=module.selector_walk_triton
    extract(WORKER,['_selector_lattice'],namespace=env)
    _,patched=patched_source();tree=ast.parse(patched)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_SelectorDraftSampler')
    astmod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    exec(compile(ast.fix_missing_locations(astmod),str(WORKER),'exec'),env)
    torch.manual_seed(113)
    sel=N(top_k=16,predecessor_codebook=torch.randn(128,8,device='cuda'),
          successor_codebook=torch.randn(128,8,device='cuda'),
          hidden_projection=nn.Linear(32,8,bias=False,device='cuda'))
    sel.build_lattice=lambda **kw:env['build_lattice'](sel,**kw)
    sel.sample_path=lambda **kw:env['sample_path'](sel,**kw)
    ids=torch.stack([torch.randperm(128,device='cuda')[:16] for _ in range(7)])[None].contiguous()
    unary=torch.randn(1,7,16,device='cuda');hidden8=torch.randn(8,32,device='cuda')
    inputids=torch.zeros(8,dtype=torch.int64,device='cuda');inputids[0]=77
    model=N(candidate_selector=sel,compute_candidates=lambda h:(ids.flatten(0,1),unary.flatten(0,1)))
    create=lambda:env['_SelectorDraftSampler'](draft_model=model,block_size=8,max_bs=1,device='cuda',sampling_enabled=False)
    baseline=create();candidate=create()
    def produce(sampler):sampler(hidden8,inputids)
    # Warm native kernels/torch operations before any graph capture.
    warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):produce(baseline);produce(candidate)
    torch.cuda.current_stream().wait_stream(warm);torch.cuda.synchronize()
    print('PHASE warmup complete',flush=True)
    original=torch.cuda.CUDAGraph()
    with torch.cuda.graph(original):produce(baseline)
    owner=ScoreExport(device='cuda',epoch=5);candidate._research_score_export=owner
    # Public phase toggles/warmup writes cannot authorize a CUDA graph.
    rogue=ScoreExport(device='cuda');rogue.begin_capture()
    sample_scores=sel.build_lattice(candidate_ids=ids,unary_logits=unary,
        hidden_states=hidden8[None,1:],anchor_token_ids=inputids[:1])
    sample_tokens=baseline.out.view(1,7)
    reject('uncaptured_write_rejected',lambda:rogue.capture_write(ids,sample_scores,sample_tokens))
    reject('unrelated_graph_binding_rejected',lambda:rogue.finish_capture(original,torch.cuda.current_stream().cuda_stream))
    rogue.abort('negative fixture retired')
    owner.capture_cuda(lambda:produce(candidate))
    print('PHASE both graphs captured',flush=True)
    report['persistent_export_bytes']=owner.out.numel()*owner.out.element_size()+owner.tokens.numel()*owner.tokens.element_size()
    pointers=(ids.data_ptr(),unary.data_ptr(),hidden8.data_ptr(),inputids.data_ptr(),owner.out.data_ptr(),owner.tokens.data_ptr())
    policy=Policy(True,1.) # Explicit synthetic criterion only; no calibration.
    generations=[]
    for generation in range(16):
        reserve('generation_'+str(generation))
        unary.normal_();hidden8.normal_();sel.hidden_projection.weight.normal_()
        ids.copy_(torch.stack([torch.randperm(128,device='cuda')[:16] for _ in range(7)])[None])
        inputids[0]=(generation*11)%128
        if generation in (0,1):
            sel.hidden_projection.weight.zero_();unary.zero_();unary[:,:,0]=.01 if generation==0 else 10.
        owner.out.fill_(float('nan'));owner.tokens.fill_(-123)
        original.replay();expected=baseline.out.clone();expected_q=baseline.q_out.clone()
        snap=owner.replay_cuda_blocking('req'+str(generation))
        score=sel.build_lattice(candidate_ids=ids,unary_logits=unary,
            hidden_states=hidden8[None,1:],anchor_token_ids=inputids[:1]).cpu()
        ci=ids.cpu();tok=expected.cpu().view(1,7);oracle=[];prev=0
        for j in range(7):
            idx=int((ci[0,j]==tok[0,j]).nonzero()[0]);row=score[0,j,prev]
            oracle.append(float(row[idx]-torch.cat((row[:idx],row[idx+1:])).max()));prev=idx
        check(torch.equal(candidate.out,expected),'native_proposals_preserved')
        check(torch.equal(candidate.q_out,expected_q),'native_q_preserved')
        check(snap.valid and snap.margins==tuple(oracle),'native_exact_path_margins')
        check(snap.proposed_tokens==tuple(tok.flatten().tolist()),'native_score_token_binding')
        check(pointers==(ids.data_ptr(),unary.data_ptr(),hidden8.data_ptr(),inputids.data_ptr(),owner.out.data_ptr(),owner.tokens.data_ptr()),'stable_pointers')
        packet=make_packet(snap,policy,snap.proposed_tokens)
        generations.append({'generation':generation,'width':packet.width,'valid':snap.valid})
    check([g['width'] for g in generations[:2]]==[4,8],'synthetic_policy_crossing')
    foreign=torch.cuda.Stream()
    with torch.cuda.stream(foreign):
        reject('actual_current_stream_not_caller_key',lambda:owner.begin_replay(owner.graph,owner.stream,'foreign'))
    # Eager CPU keys alone cannot suppress the actual consume stream check.
    generation=owner.begin_replay(owner.graph,owner.stream,'stream-test');owner.graph.replay()
    owner.finish_replay_enqueue(generation)
    with torch.cuda.stream(foreign):
        reject('actual_consumer_stream',lambda:owner.consume_blocking(generation,owner.stream,'stream-test'))
    snap=owner.consume_blocking(generation,owner.stream,'stream-test')
    check(snap.valid,'recover_same_stream_after_rejected_consumer')
    # Stale graph/pointer controls are terminal only after all healthy replays.
    reject('stale_graph',lambda:owner.begin_replay(original,owner.stream,'bad'))
    owner.tokens.data=torch.empty_like(owner.tokens)
    reject('token_data_rebinding',owner.guard)
    torch.cuda.synchronize();reserve('complete')
    report.update(status='GPU_COMPONENT_PASS',generations=generations,
      source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
          (MODEL,WORKER,kernels,P/'score_prefix_controller.py',P/'source_extract.py',P/'build_score_overlay.py',Path(__file__))},
      timing_claim='None. Correctness/lifetime gate only; explicit blocking D2H copies included in execution.')
    save()
    faulthandler.cancel_dump_traceback_later()
try:
    main()
except BaseException as e:
    report.update(status='FAILED',error=repr(e));save();raise

