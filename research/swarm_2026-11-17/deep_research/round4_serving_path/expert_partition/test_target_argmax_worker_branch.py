import ast, hashlib, json
from pathlib import Path
from types import SimpleNamespace as NS
H=Path(__file__).resolve().parent; O=H/'target_argmax_overlay/python/sglang/srt'
tree=ast.parse((O/'speculative/dflash_worker_v2.py').read_text())
branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and 'target_lp is not None and target_lp.is_deferred_shard' in ast.unparse(n.test))
code=compile(ast.Module(body=[branch],type_ignores=[]),'actual-worker-overlay-branch','exec')
results=[]
for eligible in [False,True]:
 for graph in [False,True]:
  for configured in [False,True]:
   for hit in [False,True]:
    calls=[]; logits=object(); captured=object(); original=object(); gathered=object()
    def consume(**kw):
     calls.append(('consume',kw)); return captured if kw['graph_used'] and hit else None
    lp=NS(is_deferred_shard=lambda x:x is logits,deferred_argmax=lambda x:(calls.append(('argmax',x)) or original),finish_deferred_gather=lambda x:(calls.append(('gather',x)) or gathered))
    adapter=NS(consume=consume) if configured else None
    worker=NS(_can_skip_verify_logits=lambda b,s:eligible,_research_target_argmax_capture=adapter,block_size=8)
    out=NS(next_token_logits=logits)
    scope=dict(target_lp=lp,logits_output=out,self=worker,batch=object(),sampling_info=object(),target_predict=None,bs=2,can_run_cuda_graph=graph)
    exec(code,scope)
    if not eligible:
     assert scope['target_predict'] is None and out.next_token_logits is gathered and [x[0] for x in calls]==['gather']
    else:
     success=configured and graph and hit
     assert scope['target_predict'] is (captured if success else original) and out.next_token_logits is None
     assert sum(x[0]=='argmax' for x in calls)==(0 if success else 1)
     if configured: assert calls[0][1]=={'raw_rows':16,'graph_used':graph,'can_skip_verify_logits':True}
    results.append(dict(greedy_eligible=eligible,graph=graph,configured=configured,cache_hit=hit,calls=[x[0] for x in calls]))
report={'scope':'Actual AST-extracted overlay worker branch, CPU fake outputs; checks original fallback call and sampled gather exactly once, no numerical claim','cases':results,'count':len(results),'worker_sha256':hashlib.sha256((O/'speculative/dflash_worker_v2.py').read_bytes()).hexdigest()}
(H/'TARGET_ARGMAX_WORKER_BRANCH_RESULT.json').write_text(json.dumps(report,indent=2)+'\n'); print('PASS',len(results))
