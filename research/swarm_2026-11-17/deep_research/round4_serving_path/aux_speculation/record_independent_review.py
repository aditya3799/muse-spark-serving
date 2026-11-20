import hashlib,json
from pathlib import Path
P=Path(__file__).parent;ROOT=P.parents[3];E=P.parent/'expert_partition';SRT=ROOT/'sglang/python/sglang/srt'
paths=[E/'dflash_target_argmax_adapter.py',E/'target_argmax_overlay.diff',SRT/'model_executor/model_runner.py',SRT/'model_executor/runner/decode_cuda_graph_runner.py',SRT/'managers/scheduler.py']
r={'status':'FOCUSED_SOURCE_REVIEW_NO_DECISIVE_BLOCKER','scope':'Independent startup/schema/graph ownership/stream review only; parent owns native collective gate',
 'findings':[
  'ModelRunner.capture_tail_hooks exists from __init__ line396; forward_stream created line469 before targetcapture.',
  'Scheduler builds draftworker before init_all_cuda_graphs; constructor adapter registration occurs before targetgraphcapture. getattr guard for absent decode_cuda_graph_runner is necessary and present.',
  'Capture-tail callback exists inside run_once; success notification after backend.capture_one distinguishes warmup from installedgraph.',
  'Runner fields and GraphShapeKey attributes used by admission exist in pinned source; exactrunner/backend/width/key/object checks rejectreplacedgraph orunsupportedvariants.',
  'Latestadapter preserves exactLPobject/scalarconfig and servingforwardstream key; captures use a privatestream, so bindingMRforwardstream ratherthancapturestream is appropriate.',
  'Scheduler.schedule_stream is created later for nonoverlap; thisisdistinctfromMRforwardstream. Latestargmaxfallback excludesunsupported nonoverlap stream.',
  'Abort clearsconfirmedstate; strongrefs andoutputfingerprint guard stalegeneration/mutableoutputbinding. Capacitygrowth safelyrequiresnewadapter.'
 ],
 'limits':['No independent realTP8 NCCL capturetest inthislane','No fulltargetmodelstartup or servingperformance claim','Exactbucket-only fallback retains paddedbucket eagerargmax','Non-greedy graphs stillpaycapturedargmax evenwhenconsumerfallsback','Prefixwidthoverlay rejectsexistingcapturetailhooks; combinationsneedseparateintegration'],
 'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(P/'ARGMAX_ADAPTER_INDEPENDENT_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n')
# Keep the initial audit as history, close a misleading startup-log lead explicitly.
a={'status':'INITIAL_SHORTLIST_CORRECTIONS','closed_lead':'Draft metadata glue off startup message does notproveuncaptureddeviceprep. ActualFA4 translated TARGET_VERIFY already captures normal_decode_set_metadata atmodelgraphhead. No blanketglueflagproposal.',
 'prefix_prior_art':'https://arxiv.org/html/2605.00342 documents prefixtruncation/perwidthtargetgraphs/profiledcosts; no novelty or transferredspeedup claim.',
 'remaining_secondary':'Repeatedmaskembedding collective is source-only/unmeasured andwasnotpromoted orimplemented thisround.'}
(P/'SHORTLIST_AUDIT_ADDENDUM.json').write_text(json.dumps(a,indent=2)+'\n')
print('saved independentreview andshortlistaddendum')