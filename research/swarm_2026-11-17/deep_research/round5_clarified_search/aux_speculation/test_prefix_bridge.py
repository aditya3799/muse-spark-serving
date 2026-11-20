"""Exercise real round4 resolver/configuration with explicit fake topology."""
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace as N
from dataclasses import replace
from score_prefix_controller import *
from prefix_decision_bridge import RequestPrefixBridge
P=Path(__file__).parent
PREVIOUS=P.parents[1]/'round4_serving_path/aux_speculation'
spec=importlib.util.spec_from_file_location('round4_prefix_api',PREVIOUS/'dflash_prefix_verification.py')
api=importlib.util.module_from_spec(spec);sys.modules[spec.name]=api;spec.loader.exec_module(api)
checks=0;negative=[]

def assert_(x):
    global checks
    assert x;checks+=1

def rejects(name,fn):
    try:fn()
    except RuntimeError:negative.append(name)
    else:raise AssertionError(name)

def fixture():
    backend=type('FlashAttentionBackend',(),{})();backend.fa_impl_ver=3;backend.topk=1
    runner=N(attn_backend=backend,model_config=N(is_encoder_decoder=False),lora_manager=None)
    w=N(block_size=8,target_worker=N(model_runner=runner),_need_mamba_verify_commit=False,
        lilicorr=None,_is_domino=False,_can_skip_verify_logits=lambda *a:True)
    parallel=N(attn_dp_enabled=False,attn_cp_size=1,attn_dcp_size=1,pp_size=1)
    return w,N(reqs=[N(rid='r')],sampling_info=None),parallel

policy=Policy(True,1.)
snap=Snapshot(3,1,'r',(.1,)*7,True,tuple(range(7)));packet=make_packet(snap,policy,range(7))
w,b,p=fixture();bridge=RequestPrefixBridge(api,epoch=3,policy_signature=policy.signature())
with bridge.scope(w,b,p,[packet]*8,world_size=8,request_key='r',generation=1,gather_resolved=lambda local:[local]*8) as width:
    assert_(width==4 and api.resolve_prefix_verification_width(w,b,p)==4)
assert_(not hasattr(w,'_prefix_verification_config'))

def scope_call(bridge,w,b,p,packet,generation=1):
    with bridge.scope(w,b,p,[packet]*8,world_size=8,request_key='r',generation=generation,gather_resolved=lambda local:[local]*8):pass
rejects('single_use',lambda:scope_call(bridge,w,b,p,packet))
for field,value in [('epoch',4),('generation',2),('request_key','other'),('policy_signature',('x',))]:
    bridge=RequestPrefixBridge(api,epoch=3,policy_signature=policy.signature())
    rejects('current_'+field,lambda:scope_call(bridge,w,b,p,replace(packet,**{field:value})))
bridge=RequestPrefixBridge(api,epoch=3,policy_signature=policy.signature())
b.reqs[0].rid='wrong'
rejects('actual_request_owner',lambda:scope_call(bridge,w,b,p,packet))
b.reqs[0].rid='r'
b.reqs.append(N());rejects('multi_request_no_aggregation',lambda:scope_call(bridge,w,b,p,packet))
for flag in ('lora','mamba','domino','lilicorr','attn_dp','cp','dcp','pp','nongreedy','encoder','backend'):
    w,b,p=fixture();bridge=RequestPrefixBridge(api,epoch=3,policy_signature=policy.signature())
    if flag=='lora':w.target_worker.model_runner.lora_manager=object()
    elif flag=='mamba':w._need_mamba_verify_commit=True
    elif flag=='domino':w._is_domino=True
    elif flag=='lilicorr':w.lilicorr=object()
    elif flag=='attn_dp':p.attn_dp_enabled=True
    elif flag=='cp':p.attn_cp_size=2
    elif flag=='dcp':p.attn_dcp_size=2
    elif flag=='pp':p.pp_size=2
    elif flag=='nongreedy':w._can_skip_verify_logits=lambda *a:False
    elif flag=='encoder':w.target_worker.model_runner.model_config.is_encoder_decoder=True
    elif flag=='backend':w.target_worker.model_runner.attn_backend.fa_impl_ver=4
    with bridge.scope(w,b,p,[packet],world_size=1,request_key='r',generation=1) as width:
        assert_(width==8 and w._prefix_verification_config.verify_width==8)
    assert_(not hasattr(w,'_prefix_verification_config'))
w,b,p=fixture();old=api.PrefixVerificationConfig(8,6);w._prefix_verification_config=old
bridge=RequestPrefixBridge(api,epoch=3,policy_signature=policy.signature())
try:
    with bridge.scope(w,b,p,[packet],world_size=1,request_key='r',generation=1):
        rejects('nested',lambda:scope_call(bridge,w,b,p,replace(packet,generation=2),2))
        raise ValueError('target producer failure')
except ValueError:pass
assert_(w._prefix_verification_config is old and not bridge.active)
# Reproduce review failure: local source guards disagree for identical packet.
a,ba,pa=fixture();z,bz,pz=fixture();pz.attn_cp_size=2
api.configure_prefix_verification(a,4);api.configure_prefix_verification(z,4)
assert_([api.resolve_prefix_verification_width(a,ba,pa),api.resolve_prefix_verification_width(z,bz,pz)]==[4,8])
for worker,batch,parallel in [(a,ba,pa),(z,bz,pz)]:
    bridge=RequestPrefixBridge(api,epoch=3,policy_signature=policy.signature())
    with bridge.scope(worker,batch,parallel,[packet]*2,world_size=2,request_key='r',generation=1,
                      gather_resolved=lambda local:(4,8)) as width:
        assert_(width==8)
bridge=RequestPrefixBridge(api,epoch=3,policy_signature=policy.signature())
try:
    with bridge.scope(a,ba,pa,[packet]*2,world_size=2,request_key='r',generation=1):pass
except RuntimeError:negative.append('missing_second_collective')
else:raise AssertionError('missing collective admitted')
bridge=RequestPrefixBridge(api,epoch=3,policy_signature=policy.signature())
try:
    with bridge.scope(a,ba,pa,[packet]*2,world_size=2,request_key='r',generation=1,
                      gather_resolved=lambda local:(4,0)):pass
except RuntimeError:negative.append('collective_topology_failure')
else:raise AssertionError('failed peer admitted')
result={'status':'CPU_REAL_PREFIX_RESOLVER_BRIDGE_PASS','checks':checks,'negatives':negative,
        'scope':'Actual round4 configure/resolve; fake worker topology. No target execution, CUDA graph or distributed collective.'}
(P/'CPU_PREFIX_BRIDGE_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


