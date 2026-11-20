"""CPU source-extracted dispatch/metadata/graph-lifecycle contract tests.
No torch import, CUDA context, GPU metadata kernel or native attention execution.
"""
from pathlib import Path
import types,contextlib,json,hashlib
import fa3_split_policy as pol
from source_extract import method,make_forward,FA,BASE,GRAPH
P=Path(__file__).resolve().parent;N=types.SimpleNamespace;null=contextlib.nullcontext
class Tensor:
    def __init__(self,shape=(1,),payload=None):self.shape=shape;self.dtype='int32';self.payload=payload
    def data_ptr(self):return id(self)*16
    def stride(self):return (1,)*len(self.shape)
    def contiguous(self):return self
    def view(self,*a):return self
class Mode:
    def __init__(self,name):self.name=name
    def is_target_verify(self):return self.name=='verify'
    def is_decode_or_idle(self):return self.name in ('decode','idle')
    def is_decode(self):return self.name=='decode'
    def is_idle(self):return self.name=='idle'
    def is_mixed(self):return False
    def is_draft_extend_v2(self):return self.name=='draft'
def batch(phase):return N(forward_mode=Mode(phase),batch_size=1)
checks=[];pins={}
def ok(name):checks.append({'name':name,'pass':True})
def reject(name,fn):
    try:fn()
    except pol.SplitPolicyError:ok(name);return
    raise AssertionError(name+' was admitted')
ns={'is_npu':lambda:False}
base_forward,pins['base_forward']=method(BASE,'AttentionBackend','forward',ns)
compute,pins['scheduler']=method(FA,'FlashAttentionBackend','_compute_scheduler_metadata',{})
def native_call(**kw):return kw
ext,pins['verify_call']=make_forward('verify',native_call);dec,pins['decode_call']=make_forward('decode',native_call)
class Backend:
    forward=base_forward
    _compute_scheduler_metadata=pol.fa3_scheduler_guard(compute)
    forward_extend=pol.fa3_split_scope('forward')(ext)
    forward_decode=pol.fa3_split_scope('forward')(dec)
    @pol.fa3_split_scope('metadata')
    def init_forward_metadata(self,fb):
        md=self.saved.setdefault(fb.forward_mode.name,N(page_table=Tensor(),cache_seqlens_int32=Tensor(),cu_seqlens_q=Tensor(),cu_seqlens_k=Tensor(),max_seq_len_q=8 if fb.forward_mode.is_target_verify() else 1,scheduler_metadata=None))
        if fb.forward_mode.is_decode():md.scheduler_metadata=self._compute_scheduler_metadata(1,4096,md.cache_seqlens_int32,md.cu_seqlens_q)
        self.forward_metadata=md
    @pol.fa3_split_scope('metadata')
    def stale_metadata(self,fb):
        self.forward_metadata=N(page_table=Tensor(),cache_seqlens_int32=Tensor(),cu_seqlens_q=Tensor(),max_seq_len_q=1,scheduler_metadata=Tensor(payload={'num_splits':0}))
def make_backend(enabled=True):
    b=Backend();b.num_splits=b.decode_num_splits=0;b.fa_impl_ver=3;b.is_draft_runner=False;b.use_mla=False;b._disable_scheduler_metadata_precompute=False
    b.head_dim=64;b.num_attention_heads=16;b.num_kv_heads=2;b.kv_cache_dtype='float8';b.page_size=16;b.has_softcap=False
    b._get_scheduler_metadata=lambda **kw:Tensor(payload=kw);b.saved={};b.forward_metadata=None;b.key_cache=Tensor();b.value_cache=Tensor()
    p=pol.TargetFA3SplitPolicy(verify=4,decode=2,enabled=True,collective_setup_agreed=True,precision_binary_audited=True);owner=N(_research_fa3_split_policy=p,is_draft_worker=False)
    if enabled:b._research_fa3_controller=pol.SplitController(b,owner,p)
    return b,owner
layer=N(tp_q_head_num=16,head_dim=64,v_head_dim=64,scaling=.125,logit_cap=0.)
def call(b,phase):return b.forward(Tensor(),None,None,layer,batch(phase))
b,owner=make_backend()
for phase,S in [('verify',4),('decode',2),('other',0),('verify',4),('decode',2)]:
    b.init_forward_metadata(batch(phase));r=call(b,phase);assert r['num_splits']==S
    if phase=='decode':assert r['scheduler_metadata'].payload['num_splits']==S
    assert (b.num_splits,b.decode_num_splits)==(0,0);ok('actual source '+phase+' split '+str(S))
reject('verify metadata cannot feed decode',lambda:call(b,'verify'))
reject('stale metadata without coherent producer',lambda:b.stale_metadata(batch('decode')))
b.init_forward_metadata(batch('decode'));b.forward_metadata.scheduler_metadata=Tensor();reject('changed scheduler pointer',lambda:call(b,'decode'))
b.init_forward_metadata(batch('decode'));b.decode_num_splits=8;reject('decode-only field mutation',lambda:call(b,'decode'));b.decode_num_splits=0
# An unrelated scheduler call must not qualify a different stale metadata tensor.
b,owner=make_backend();c=pol.controller(b)
with c.scope('decode'):
    before=c.scheduler_serial;produced=b._compute_scheduler_metadata(1,4096,Tensor(),Tensor())
    b.forward_metadata=N(scheduler_metadata=Tensor())
    reject('unrelated scheduler result cannot bless stale metadata',lambda:c.stamp('decode',before))
    b.forward_metadata.scheduler_metadata=produced;c.stamp('decode',before);ok('direct producer span accepted')
    buf=Tensor((8,));view=Tensor(produced.shape);view.data_ptr=buf.data_ptr
    pol.record_scheduler_copy(b,produced,buf,produced.shape[0]);b.forward_metadata.scheduler_metadata=view;c.stamp('decode',before);ok('explicit pinned prefix copy receipt accepted')
    view.data_ptr=lambda:buf.data_ptr()+4
    reject('offset scheduler view rejected',lambda:c.stamp('decode',before))
    view.data_ptr=buf.data_ptr
    reject('unrelated copy source rejected',lambda:pol.record_scheduler_copy(b,Tensor(),buf,1))
reject('out-of-phase scheduler metadata',lambda:b._compute_scheduler_metadata(1,4096,Tensor(),Tensor()))
b0,_=make_backend(False);b0.init_forward_metadata(batch('verify'));assert call(b0,'verify')['num_splits']==0;ok('default off original verify')
b0.is_draft_runner=True;b0.fa_impl_ver=4;assert call(b0,'verify')['num_splits']==0;ok('unconfigured draft FA4 unchanged')
# Negative control: a naive decode-only policy reproduces the actual source mismatch.
nb,no=make_backend();c=pol.controller(nb);orig=c.selected;c.selected=lambda phase:(0,2) if phase=='decode' else orig(phase)
nb.init_forward_metadata(batch('decode'));r=call(nb,'decode');assert r['num_splits']==2 and r['scheduler_metadata'].payload['num_splits']==0;ok('negative control catches naive decode-vs-scheduler mismatch')

class Graph:
    def __init__(self):self.replays=0;self.reset_called=False
    def replay(self):self.replays+=1
    def reset(self):self.reset_called=True
fake_torch=N(cuda=N(CUDAGraph=Graph))
gns={'torch':fake_torch,'graph_pool_capture_scope':null,'graph_pool_replay_scope':null}
methods={}
for name in ['capture_one','can_run','replay','cleanup']:
    methods[name],pins['graph_'+name]=method(GRAPH,'FullCudaGraphBackend',name,gns)
FullCudaGraphBackend=type('FullCudaGraphBackend',(),{name:decor(methods[name]) for name,decor in [('capture_one',pol.fa3_graph_capture_guard),('can_run',pol.fa3_graph_can_run_guard),('replay',pol.fa3_graph_replay_guard),('cleanup',pol.fa3_graph_cleanup_guard)]})
def graph_fixture():
    b,o=make_backend();runner=N(model_runner=o,_replay_attn_backend=lambda:b,enable_pdmux=False,enable_two_batch_overlap=False,ragged_verify_mode=False,_metadata_glue=None,enable_profile_cuda_graph=False)
    g=FullCudaGraphBackend();runner.backend=g;g._cuda_graph_runner=runner;g._graphs={};g._outputs={};g._device_module=N(synchronize=lambda:None,graph=lambda **kw:null());g._tp_group=N(barrier=lambda:None);g._precarve=N(measure=null,mint=lambda:None);g._reuse_output_buffer=False;g._memory_saver_adapter=None;g._pool=None;g._capture_stream=None;g._output_buffer=None
    return b,o,g
b,o,g=graph_fixture()
for phase in ['verify','decode']:
    b.init_forward_metadata(batch(phase));g.capture_one(phase,lambda:call(b,phase));assert g.can_run(batch(phase),phase);out=g.replay(phase,batch(phase));assert out['num_splits']==(4 if phase=='verify' else 2);ok('source full graph capture/replay '+phase)
b.init_forward_metadata(batch('verify'));g.replay('verify',batch('verify'));ok('return to original graph metadata after another bucket')
old=g._graphs['verify'];g._graphs['verify']=Graph();reject('graph identity replaced',lambda:g.replay('verify',batch('verify')));g._graphs['verify']=old
b.init_forward_metadata(batch('verify'));g.capture_one('verify',lambda:call(b,'verify'));assert g._graphs['verify'] is not old;g.replay('verify',batch('verify'));ok('successful same-policy recapture binds new graph')
g.cleanup();assert not g.can_run(batch('verify'),'verify');reject('cleanup invalidates replay',lambda:g.replay('verify',batch('verify')))
b,o,g=graph_fixture();b.init_forward_metadata(batch('verify'));g.capture_one('verify',lambda:call(b,'verify'));count=[0]
def fail_capture():
    count[0]+=1
    if count[0]==3:raise ValueError('injected native capture failure after two warmups')
    return call(b,'verify')
try:g.capture_one('verify',fail_capture)
except ValueError:pass
else:raise AssertionError('negative did not fail')
assert not g.can_run(batch('verify'),'verify');reject('failed recapture cannot revive old graph',lambda:g.replay('verify',batch('verify')));reject('poisoned backend rejects eager fallback',lambda:b.init_forward_metadata(batch('verify')))
for attr,value in [('enable_pdmux',True),('enable_two_batch_overlap',True),('ragged_verify_mode',True),('_metadata_glue',object())]:
 b,o,g=graph_fixture();setattr(g._cuda_graph_runner,attr,value);b.init_forward_metadata(batch('verify'));reject('unsupported '+attr,lambda:g.capture_one('verify',lambda:None))
b,o=make_backend();o._research_fa3_split_policy=pol.TargetFA3SplitPolicy(enabled=True);reject('policy replacement',lambda:b.init_forward_metadata(batch('verify')))
# FullCudaGraphBackend is shared with prefill runners lacking decode's resolver.
generic=N(_cuda_graph_runner=N(model_runner=N()))
wrapped=pol.fa3_graph_replay_guard(lambda self,key,batch:'unchanged')
assert wrapped(generic,1,None)=='unchanged';ok('default-off nondecode graph runner unchanged')
generic._cuda_graph_runner.model_runner._research_fa3_split_policy=pol.TargetFA3SplitPolicy(enabled=True)
reject('enabled unreviewed nondecode graph runner rejected',lambda:wrapped(generic,1,None))
b,o=make_backend();b.init_forward_metadata(batch('decode'));layer.is_swa=True;layer.window_size=(2047,0)
r=call(b,'decode');assert r['num_splits']==2 and r['scheduler_metadata'] is None and r['window_size']==(2047,0);ok('SWA decode uses coherent split and no global scheduler metadata')
layer.is_swa=False;layer.window_size=(-1,-1)
report={'status':'CPU_SOURCE_EXTRACTED_CONTRACT_PASS','checks':checks,'count':len(checks),'source_fragments_sha256':pins,'support_sha256':hashlib.sha256((P/'fa3_split_policy.py').read_bytes()).hexdigest(),'scope':'Actual pinned base dispatch, FA3 call expressions/scheduler method and complete FullCudaGraphBackend capture/replay/cleanup methods. Tensor/CUDA/group/allocator actions mocked; no native precision, key coverage or runtime qualification.'}
(P/'CPU_POLICY_TESTS.json').write_text(json.dumps(report,indent=2));print(report['status'],len(checks))
