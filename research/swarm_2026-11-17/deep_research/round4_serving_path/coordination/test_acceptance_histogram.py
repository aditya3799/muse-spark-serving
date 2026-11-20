from __future__ import annotations
import ast,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from acceptance_histogram import AcceptanceHistogram
H=Path(__file__).resolve().parent;ROOT=next(p for p in H.parents if (p/'RULES.md').exists());SRT=ROOT/'sglang/python/sglang/srt'
path=SRT/'managers/scheduler_components/batch_result_processor.py'
tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(getattr(x,'name','')=='_resolve_spec_v2_tokens' for x in n.body))
method=next(n for n in cls.body if getattr(n,'name','')=='_resolve_spec_v2_tokens')
stride=next(n for n in tree.body if getattr(n,'name','')=='_get_speculative_output_stride')
ns={};exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+[stride,method],type_ignores=[]),str(path),'exec'),ns)
bp=SRT/'speculative/base_spec_worker.py';bt=ast.parse(bp.read_text());bc=next(n for n in bt.body if isinstance(n,ast.ClassDef) and n.name=='BaseSpecWorker');callback=next(n for n in bc.body if getattr(n,'name','')=='on_verify_complete_cpu')
exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+[callback],type_ignores=[]),str(bp),'exec'),ns)
Worker=type('DFlashWorkerV2',(),{'on_verify_complete_cpu':ns['on_verify_complete_cpu']})
class HostTensor:
    is_cpu=True
    def __init__(self,values):self.values=values;self.reads=0
    def tolist(self):self.reads+=1;return list(self.values)
    def cpu(self):raise AssertionError('extra CPU copy forbidden')
    def cuda(self):raise AssertionError('CUDA use forbidden')
def fixture(counts):
    worker=Worker();worker.block_size=8
    proc=NS(model_worker=worker,metrics_reporter=NS(snapshot_spec_decode_metrics=lambda *args:None),advance_grammar_fsm=lambda *args:None)
    reqs=[NS(is_retracted=False,finished=lambda:False,grammar=None,kv=NS(kv_committed_len=10),spec_verify_ct=0,spec_num_correct_drafts=0,update_spec_correct_drafts_histogram=lambda n:None) for _ in counts]
    result=NS(next_token_ids=HostTensor(list(range(len(counts)*8))),accept_lens=HostTensor([c+1 for c in counts]),
        speculative_output_stride=8,speculative_num_draft_tokens=8,num_non_draft_tokens_per_req=1,block_accept_lens=None,cap_lens=None)
    return worker,proc,NS(reqs=reqs),result
checks=[]
for counts in ([0],[7],[0,4,1],[7,7,7],[0,1,2,3,4,5,6,7]):
    w,p,b,r=fixture(counts);expected=ns['_resolve_spec_v2_tokens'](p,r,b)
    w,p,b,r=fixture(counts);observer=AcceptanceHistogram().install(w)
    actual=ns['_resolve_spec_v2_tokens'](p,r,b)
    assert actual==expected
    assert r.next_token_ids.reads==1 and r.accept_lens.reads==1
    row=observer.rows[len(counts)]
    assert row['correct_draft_histogram']==[counts.count(i) for i in range(8)]
    assert all(hist[sum(c>=width for c in counts)]==1 for width,hist in row['survivor_count_histograms'].items())
    observer.uninstall();assert 'on_verify_complete_cpu' not in vars(w)
    checks.append({'counts':counts,'original_outputs_preserved':True,'no_extra_tensor_reads_or_copies':True})
# Same marginal histogram, different per-batch survivor correlation.
a=AcceptanceHistogram();b=AcceptanceHistogram()
a.record([0,4],2);a.record([0,4],2);b.record([0,0],2);b.record([4,4],2)
assert a.rows[2]['correct_draft_histogram']==b.rows[2]['correct_draft_histogram']
assert a.rows[2]['survivor_count_histograms'][4]!=b.rows[2]['survivor_count_histograms'][4]
c=AcceptanceHistogram(every=2,max_steps=2,max_batch=4)
for _ in range(8):c.record([1,2],2)
assert c.sampled==2
bad=AcceptanceHistogram()
for values,bs in [([8],1),([-1],1),([True],1),([1],2),((1,),1),([],0)]:bad.record(values,bs)
assert bad.invalid==6 and not bad.rows
w,p,batch,r=fixture([1]);sentinel=object();seen=[]
def old(values,batch_size=0):seen.append((list(values),batch_size));return sentinel
w.on_verify_complete_cpu=old;observer=AcceptanceHistogram().install(w)
assert w.on_verify_complete_cpu([1],batch_size=1) is sentinel and seen==[([1],1)]
observer.uninstall();assert w.on_verify_complete_cpu is old
assert 'torch' not in sys.modules
report={'status':'PASS_ACTUAL_CPU_CALLBACK_EXTRACTION','source_cases':checks,'joint_survivor_correlation_distinguished':True,
        'sampling_bounds_checked':True,'invalid_schema_cases':6,'callback_return_and_uninstall_preserved':True,'torch_imported':False,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,bp,H/'acceptance_histogram.py',Path(__file__).resolve()]},
        'limits':['Uses extracted real resultprocessor/callback with fake CPU tensor interfaces','No active installation, real request stream or timing evidence']}
(H/'ACCEPTANCE_HISTOGRAM_CPU_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'source_cases':len(checks),'torch_imported':False}))
