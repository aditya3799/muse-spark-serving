import ast,contextlib,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace as NS
P=Path(__file__).resolve().parent
source=P/'profile_humming_stages.py';tree=ast.parse(source.read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
selected=[n for n in main.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Recorder','hook','hooks')]
class Event:
 def __init__(self,**kw):self.flags=kw;self.records=0
 def record(self):self.records+=1
 def elapsed_time(self,other):return .001
class Humming:
 @classmethod
 def forward_layer(cls,layer,**kw):return cls,layer,kw
origin=Humming.forward_layer.__func__
ns={'torch':NS(cuda=NS(Event=Event,synchronize=lambda:None),profiler=NS(record_function=lambda name:contextlib.nullcontext())),'time':time,'collections':__import__('collections'),'originals':[],'HummingMethod':Humming}
exec(compile(ast.Module(body=selected,type_ignores=[]),'exact_profiler_nodes','exec'),ns)
rec=ns['Recorder']();ns['rec']=rec
checks=[]
def check(label,ok):assert ok,label;checks.append(label)
ns['hook'](Humming,'forward_layer',lambda kw:'gemm_'+kw['sublayer_name'])
other=NS(f=lambda x:x+3);origf=other.f;ns['hook'](other,'f','ordinary')
ns['hooks'](False)
check('restore_classmethod_binding',Humming.forward_layer.__func__ is origin and Humming.forward_layer(17,sublayer_name='w13')==(Humming,17,{'sublayer_name':'w13'}))
check('restore_plain_function',other.f is origf)
ns['hooks'](True);rec.reset('events')
check('hook_return_identity',Humming.forward_layer(17,sublayer_name='w13')==(Humming,17,{'sublayer_name':'w13'}))
check('ordinary_hook_return',other.f(8)==11)
check('stage_order',[r[0] for r in rec.records]==['gemm_w13','ordinary'])
check('record_event_pairs',rec.records[0][1].records==2 and rec.records[0][2].records==2)
r=rec.summarize(rec.records,host_valid=False)
check('capture_cpu_not_reported',all(v['host_us'] is None for v in r['calls']) and all(v['host_us']==0 for v in r['stages'].values()))
check('stream_time_sum',r['total_stage_stream_us']==2)
rec.reset('profiler');check('profiler_result',other.f(2)==5);check('profiler_no_event_records',len(rec.records)==0)
rec.reset(None);check('disabled_hook_no_records',other.f(9)==12 and len(rec.records)==0)
for _ in range(3):ns['hooks'](False);ns['hooks'](True)
ns['hooks'](False)
check('repeated_restore_preserves_identity',Humming.forward_layer.__func__ is origin and other.f is origf)
rec.reset('events');ns['hooks'](True)
for _ in range(32):other.f(0)
try:other.f(0);overflow=False
except AssertionError:overflow=True
check('bounded_stage_event_pool',overflow)
rec.reset('events');check('reset_event_cursor',rec.cursor==0 and not rec.records)
# Plain capture and second controls explicitly restore methods before enqueue.
text=source.read_text();check('plain_capture_restores_methods','rec.mode=None;hooks(False)\n        graph,graph_output=capture' in text)
check('second_plain_controls_restore_methods','hooks(False)\n        eager_after' in text)
report={'status':'PASS','checks':len(checks),'names':checks,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Exact Python hook/event bookkeeping extracted AST with CPU spies; no CUDA, external-event capture or profiler runtime proof.'}
(P/'CPU_PROFILER_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','checks':len(checks)}))
