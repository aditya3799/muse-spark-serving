"""Capture-stable control and fixed8 SGLang boundary CPU contract."""
import hashlib,json
from pathlib import Path
from full8_graph_protocol import Fixed8PrefixController
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
checks=0;negative=[]
for bs in (1,4,16,32):
    ctl=Fixed8PrefixController(enabled=True);control=ctl.device_width
    # A single captured callable loads a mutable device control at replay time.
    def captured_target(inputs):
        width=inputs['mask_control'].load()
        return [[c+1 for c in row] for row in inputs['tokens']],[[('real',i,j) if j<width else ('unverified',i,j) for j in range(8)] for i in range(bs)]
    graph_identity=id(captured_target)
    for width in (8,2,4,6,1,8,4):
        ctl.set_width(width)
        tokens=[[100+i*100+j for j in range(8)] for i in range(bs)]
        pos=[[30+i*20+j for j in range(8)] for i in range(bs)]
        slots=[[9000+i*128+j for j in range(8)] for i in range(bs)]
        args=ctl.begin(tokens,pos,slots)
        assert args['draft_token_num']==args['num_tokens_per_req']==args['output_stride']==8
        assert args['mask_control'] is control and id(captured_target)==graph_identity
        predictions,hidden=captured_target(args)
        commits=ctl.accept(predictions,[row[0] for row in pos])
        assert all(c.committed==width and c.bonus==tokens[i][width-1]+1 for i,c in enumerate(commits))
        mat=ctl.materialize(hidden,args['generation'])
        assert mat['full_hidden'] is hidden and all(len(row)==8 for row in mat['full_hidden'])
        assert mat['full_positions'] is pos and mat['full_cache_slots'] is slots
        assert all(len(row)==width and all(x[0]=='real' for x in row) for row in mat['written_hidden'])
        ctl.end();checks+=1
    ctl.set_width(4);args=ctl.begin(tokens,pos,slots)
    for name,action in [('control_changed_inflight',lambda:ctl.set_width(2)),('next_target_before_aux',lambda:ctl.begin(tokens,pos,slots)),('release_before_aux',ctl.end),('stale_aux',lambda:ctl.materialize(hidden,args['generation']-1))]:
        try:action()
        except RuntimeError:negative.append(name)
        else:raise AssertionError(name)
    predictions,hidden=captured_target(args);ctl.accept(predictions,[row[0] for row in pos]);ctl.materialize(hidden,args['generation']);ctl.end()
# Exact batch lengths, atomic failure and duplicate acceptance are required.
for mode in ('extra_targets','missing_targets','extra_lengths','missing_lengths','bad_row','bad_prefix_value'):
    ctl=Fixed8PrefixController(enabled=True);ctl.set_width(4)
    tok=[[10+j for j in range(8)],[30+j for j in range(8)]]
    args=ctl.begin(tok,tok,tok);pred=[[x+1 for x in row] for row in tok];lengths=[5,9]
    if mode=='extra_targets':pred=pred+[pred[0]]
    if mode=='missing_targets':pred=pred[:1]
    if mode=='extra_lengths':lengths=lengths+[7]
    if mode=='missing_lengths':lengths=lengths[:1]
    if mode=='bad_row':pred[1]=pred[1][:-1]
    if mode=='bad_prefix_value':lengths[1]=object()
    try:ctl.accept(pred,lengths)
    except (ValueError,TypeError):negative.append(mode)
    else:raise AssertionError(mode)
    assert not ctl.accepted and not hasattr(ctl,'commits')
    try:ctl.materialize(tok,args['generation'])
    except RuntimeError:negative.append('materialize_after_failed_'+mode)
    else:raise AssertionError('partial acceptance was published')
    good=ctl.accept([[x+1 for x in row] for row in tok],[5,9])
    try:ctl.accept([[999]*8]*2,[5,9])
    except RuntimeError:negative.append('duplicate_acceptance')
    else:raise AssertionError('duplicate acceptance')
    assert ctl.commits is good
    ctl.materialize(tok,args['generation']);ctl.end()
ctl=Fixed8PrefixController();ctl.set_width(4);assert ctl.device_width.load()==8
files=['sglang/python/sglang/srt/speculative/dflash_worker_v2.py','sglang/python/sglang/srt/speculative/dflash_info.py','sglang/python/sglang/kernels/ops/kvcache/cache_move.py','sglang/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py']
report={'status':'PASS','fixed_capture_shape_width_transitions':checks,'negative_lifetime_guards':negative,
        'default_width':8,'new_target_graph_captures_required_by_contract':0,
        'scope':'CPU capture-control and source boundary contract; actual CUDA scalar registry, TP scatter mapping, worker/router integration and H200 fullmodel remain unimplemented/unproven.',
        'pinned_sources':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(P/'cpu_full8_graph_contract_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
