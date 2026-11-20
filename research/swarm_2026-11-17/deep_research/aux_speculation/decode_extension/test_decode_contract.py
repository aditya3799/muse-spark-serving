"""Source-extracted decode allocation/capture/replay + canonical-buffer lease."""
import argparse,ast,contextlib,copy,dataclasses,enum,importlib.util,json,sys,types
from pathlib import Path
from types import SimpleNamespace,MethodType
from unittest.mock import patch
from typing import *
import torch
P=Path(__file__).resolve().parent;A=P.parent;S=A.parents[2]/'sglang/python/sglang';C=P/'direct_sources/python/sglang'
pa=argparse.ArgumentParser();pa.add_argument('--device',default='cpu');pa.add_argument('--hidden',type=int,default=64);pa.add_argument('--fused',action='store_true');pa.add_argument('--hot',action='store_true');pa.add_argument('--out',type=Path,required=True);a=pa.parse_args();checks=[];H=a.hidden
if a.fused:C=P/'fused_sources/python/sglang'
if a.hot:C=P/('hot_fused_sources/python/sglang' if a.fused else 'hot_direct_sources/python/sglang')
def check(name,ok,**kw):assert ok,name;checks.append({'name':name,'passed':bool(ok),**kw})
def extract(path,names,env=None,cls=None):
    tree=ast.parse(path.read_text());body=tree.body
    if cls:body=next(n for n in body if isinstance(n,ast.ClassDef) and n.name==cls).body
    nodes=[copy.deepcopy(n) for n in body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names),(names,len(nodes))
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[]);ast.fix_missing_locations(mod)
    module=types.ModuleType('decode_fixture_'+str(len(sys.modules)));sys.modules[module.__name__]=module
    module.__dict__.update(torch=torch,dataclasses=dataclasses,dataclass=dataclasses.dataclass,fields=dataclasses.fields,Enum=enum.Enum,IntEnum=enum.IntEnum,auto=enum.auto,**(env or {}))
    exec(compile(mod,str(path),'exec'),module.__dict__);return {n:getattr(module,n) for n in names}
def method(path,cls,name,env=None):return extract(path,[name],env,cls)[name]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
baseowner=load('sglang.srt.layers.prefill_aux_output',C/'srt/layers/prefill_aux_output.py')
decode=load('decode_binding',C/'srt/layers/decode_aux_output.py')
mode=extract(S/'srt/model_executor/forward_batch_info.py',['ForwardMode'])['ForwardMode']
check('real_target_verify_is_extend',mode.TARGET_VERIFY.is_extend())
parallel=SimpleNamespace(pp_size=1,attn_dp_enabled=False,attn_cp_size=1)
execution=SimpleNamespace(overlap=SimpleNamespace(enable_two_batch_overlap=False));disagg=SimpleNamespace(enable_pdmux=False)
modelcap=extract(C/'srt/models/muse_spark.py',['decode_aux_output_spec','get_aux_hidden_states_width'],{'get_parallel':lambda:parallel,'get_exec':lambda:execution,'get_disagg':lambda:disagg},'MuseSparkTextForCausalLM')
spec=(H,(0,2,4,6,8));outer=SimpleNamespace(capture_aux_hidden_states=True,model=SimpleNamespace(prefill_aux_output_spec=lambda:spec))
outer.decode_aux_output_spec=lambda:modelcap['decode_aux_output_spec'](outer)
check('capability_width',modelcap['get_aux_hidden_states_width'](outer)==H*5)
outer.get_aux_hidden_states_width=lambda:modelcap['get_aux_hidden_states_width'](outer)
resolver=extract(S/'srt/model_executor/model_runner_components/misc_utils.py',['_SupportsPackedAuxHiddenStates','resolve_aux_hidden_states_width'],{'Protocol':Protocol,'runtime_checkable':runtime_checkable})['resolve_aux_hidden_states_width']
for dflash,draft in [(True,False),(False,False),(True,True)]:
    width=resolver(model=outer,spec_algorithm=SimpleNamespace(is_dflash_family=lambda:dflash),is_draft_worker=draft)
    check('actual_resolver_%s_%s'%(dflash,draft),width==(H*5 if dflash and not draft else 0))

for name,obj,attr,value in [('pp',parallel,'pp_size',2),('dp',parallel,'attn_dp_enabled',True),('cp',parallel,'attn_cp_size',2),('tbo',execution.overlap,'enable_two_batch_overlap',True),('pdmux',disagg,'enable_pdmux',True),('capture_off',outer,'capture_aux_hidden_states',False)]:
    before=getattr(obj,attr);setattr(obj,attr,value);check('capability_guard_'+name,modelcap['get_aux_hidden_states_width'](outer)==0);setattr(obj,attr,before)
# Actual dataclass creation and process-wide canonical-buffer sharing.
pool={};ib=extract(S/'srt/model_executor/input_buffers.py',['share_input_buffer','ForwardInputBuffers'],{'_forward_input_buffer_pool':pool,'is_npu':lambda:False})
envs=SimpleNamespace(SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE=SimpleNamespace(get=lambda:False),SGLANG_LOG_DECODE_GRAPH_KEY=SimpleNamespace(get=lambda:False))
Buffers=extract(S/'srt/model_executor/runner_utils/buffers.py',['DecodeInputBuffers'],{'ForwardInputBuffers':ib['ForwardInputBuffers'],'enable_num_token_non_padded':lambda:False,'envs':envs})['DecodeInputBuffers']
def create(rows=64):return Buffers.create(device=torch.device(a.device),max_bs=rows//8,max_num_token=rows,hidden_size=H,next_token_logits_buffer=torch.empty(rows,17,dtype=torch.bfloat16,device=a.device),dtype=torch.bfloat16,num_dp_ranks=1,pp_size=1,is_encoder_decoder=False,require_mlp_tp_gather=False,seq_len_fill_value=8,encoder_len_fill_value=0,num_tokens_per_req=8,cache_loc_dtype=torch.int64,enable_mamba_track=False,aux_hidden_states_width=H*5)
buffers=create();buffers.share_buffers();alias=create();alias.share_buffers();other=create(32);other.share_buffers()
check('actual_decode_buffer_shape',buffers.aux_hidden_states.shape==(64,H*5))
check('canonical_same_capacity_alias',buffers.aux_hidden_states.data_ptr()==alias.aux_hidden_states.data_ptr())
check('canonical_different_capacity_separate',buffers.aux_hidden_states.data_ptr()!=other.aux_hidden_states.data_ptr())
class FakeStream:
    def __init__(self,name):self.name=name
    def wait_stream(self,other):pass
stream=FakeStream('cpu')
ctx=contextlib.ExitStack() if a.device=='cpu' else contextlib.nullcontext()
with ctx as stack:
    if a.device=='cpu':stack.enter_context(patch.object(torch.cuda,'current_stream',lambda device=None:stream))
    lease=decode.DecodeAuxOutputLease(buffers.aux_hidden_states,H,spec[1])
    check('lease_no_payload_allocation',lease.buffer is buffers.aux_hidden_states)
    lease.validate_binding(buffers.aux_hidden_states,spec)
    for reason,buf,sp in [('pointer',buffers.aux_hidden_states.clone(),spec),('rows',buffers.aux_hidden_states[:32],spec),('ids',buffers.aux_hidden_states,(H,(1,2,4,6,8))),('dtype',buffers.aux_hidden_states.float(),spec)]:
        try:lease.validate_binding(buf,sp)
        except RuntimeError:check('binding_reject_'+reason,True)
        else:check('binding_reject_'+reason,False)
    # Preserve original live-Tensor fingerprint failure as an explicit negative control.
    if a.hot:
        original_helper=load('original_live_binding',P/'direct_sources/python/sglang/srt/layers/decode_aux_output.py')
        negative_buffer=torch.empty_like(buffers.aux_hidden_states)
        negative=original_helper.DecodeAuxOutputLease(negative_buffer,H,spec[1])
        old_pointer=negative_buffer.data_ptr();negative_buffer.data=negative_buffer.clone()
        negative.validate_binding(negative_buffer,spec)
        check('negative_original_misses_inplace_storage_change',negative_buffer.data_ptr()!=old_pointer)
        for reason in ['storage','shape','stride','dtype']:
            tensor=torch.empty_like(buffers.aux_hidden_states);binding=decode.DecodeAuxOutputLease(tensor,H,spec[1]);original_view=tensor.view_as(tensor)
            if reason=='storage':tensor.data=tensor.clone()
            elif reason=='shape':tensor.resize_(32,H*5)
            elif reason=='stride':tensor.as_strided_((64,H*5),(1,64))
            else:tensor.data=tensor.float()
            try:binding.validate_binding(tensor,spec)
            except RuntimeError:check('hot_reject_inplace_'+reason,True)
            else:check('hot_reject_inplace_'+reason,False)
            tensor.data=original_view
            binding.validate_binding(tensor,spec)
            generation=binding.begin_produce();binding.end_produce(generation,binding.view(8))
            with binding.consume(binding.view(8)):pass
            check('hot_recovery_after_restore_'+reason,binding._state=='idle')

    @dataclasses.dataclass
    class Logits:
        next_token_logits:object=None
        hidden_states:object=None
        last_hidden_states:object=None
        full_logits:object=None
        marker:str='preserved'
    class ReadEnds(enum.Enum):PRE_REPLAY=1;IN_REPLAY=2;POST_REPLAY=3;NONE=4
    renv={'DecodeAuxOutputLease':decode.DecodeAuxOutputLease,'get_parallel':lambda:parallel,'ForwardBatch':SimpleNamespace,'DpPaddingMode':SimpleNamespace(get_default_mode_in_cuda_graph=lambda n:None),'get_required_capture_hidden_mode':lambda *args:'full','enable_num_token_non_padded':lambda:False,'device_timer_ctx':lambda *args:contextlib.nullcontext(),'SharedReadEnds':ReadEnds,'envs':envs,'LogitsProcessorOutput':Logits,'PPProxyTensors':type('PP',(),{})}
    runner_path=C/'srt/model_executor/runner/decode_cuda_graph_runner.py'
    names=['_init_decode_aux_output_lease','_validate_decode_aux_binding','_decode_aux_capture_buffer','adopt_owned_cuda_graph_output','capture_prepare','capture','execute','_execute_with_aux_output']
    rm=extract(runner_path,names,renv,'DecodeCudaGraphRunner')
    class Slot:
        def __init__(self,name):self.name=name
        def slice_for(self,bs,tokens):
            x=getattr(buffers,self.name)
            return x[:,:tokens] if self.name=='mrope_positions' else x[:bs] if self.name in ['req_pool_indices','seq_lens','seq_lens_cpu'] else x[:tokens]
    registry=SimpleNamespace(get_slot=lambda name:Slot(name),has_slot=lambda name:False)
    mr=SimpleNamespace(model=outer,is_draft_worker=False,spec_algorithm=SimpleNamespace(is_dflash=lambda:True),attn_tp_sequence_sharded=lambda n:False,lora_manager=None,kv_index_translator=SimpleNamespace(bind_runner_slots=lambda fb:None),hisparse_coordinator=None,device_timer=None)
    r=SimpleNamespace(model_runner=mr,buffers=buffers,buffer_registry=registry,decode_aux_output_lease=lease,captured_req_width=8,pp_size=1,enable_prefill_cp=False,enable_pdmux=False,enable_two_batch_overlap=False,device=a.device,attn_backend=None,capture_forward_mode=mode.TARGET_VERIFY,capture_hidden_mode='full',get_spec_info=lambda n:None,_global_num_tokens_for_graph=lambda n:None,model_context=contextlib.nullcontext,is_dllm=False,_resolve_shared_read_ends=lambda *args:ReadEnds.NONE,_replay_attn_backend=lambda:None,_process_output_after_replay=lambda out,fb:out)
    for name,fn in rm.items():setattr(r,name,MethodType(fn,r))
    # Real initializer only binds the pre-existing tensor after all topology/mode guards.
    r.decode_aux_output_lease=None;r.device='cuda';r._init_decode_aux_output_lease()
    check('actual_initializer_device_binding',r.decode_aux_output_lease.buffer is buffers.aux_hidden_states if a.device!='cpu' else r.decode_aux_output_lease is None)
    for reason,obj,attr,value in [('draft',mr,'is_draft_worker',True),('pp',r,'pp_size',2),('pdmux',r,'enable_pdmux',True),('tbo',r,'enable_two_batch_overlap',True),('device',r,'device','cpu'),('dp',parallel,'attn_dp_enabled',True),('cp',parallel,'attn_cp_size',2),('mode',r,'capture_forward_mode',mode.DECODE)]:
        before=getattr(obj,attr);setattr(obj,attr,value);r.decode_aux_output_lease=None;r._init_decode_aux_output_lease()
        check('actual_initializer_guard_'+reason,r.decode_aux_output_lease is None);setattr(obj,attr,before)
    r.decode_aux_output_lease=lease;r.device=a.device

    for bs in [8,1,3,8]:
        fb,_,_=r.capture_prepare(bs)
        check('actual_capture_bind_bs%d'%bs,fb.aux_hidden_states_buffer.shape==(bs*8,H*5) and fb.aux_hidden_states_buffer.data_ptr()==lease.buffer.data_ptr())
    previous=lease.generation;r.stream=torch.cuda.current_stream();r._capture=lambda:r.capture_prepare(8);r.capture();check('actual_capture_generation',lease.generation==previous+1 and lease._state=='idle')
    # Source model forward with in-place fixture residual arithmetic.
    class Layer:
        def __init__(self,i):self.i=i;self.self_attn_input_norm=lambda x:x
        def __call__(self,pos,x,attn,fb,aux_out=None,aux_slot=0):
            x.add_((self.i+1)*.03125);x.mul_(1.015625)
            if aux_out is not None:aux_out[:,aux_slot*H:(aux_slot+1)*H].copy_(x)
            return x,x
    model=SimpleNamespace(embed_norm=None,layers=[Layer(i) for i in range(9)],layers_to_capture=list(spec[1]),residual_dtype=torch.float32,operator_dtype=torch.bfloat16,norm=lambda x:x)
    menv={'direct_aux_capture':baseowner.direct_aux_capture,'get_forward':lambda:SimpleNamespace(set=lambda *args:None),'layernorm_sp':SimpleNamespace(runs_sp=lambda mode:False)}
    forward=method(C/'srt/models/muse_spark.py','MuseSpark1xModel','forward',menv)
    old=method(A/'base_sources/python/sglang/srt/models/muse_spark.py','MuseSpark1xModel','forward',menv)
    staticfb,_,_=r.capture_prepare(8);inputs=torch.empty(64,H,dtype=torch.bfloat16,device=a.device)
    def replay(key,fb):
        hidden,aux=forward(model,None,None,staticfb,inputs)
        return Logits(next_token_logits=torch.empty(64,17,device=a.device),hidden_states=aux,last_hidden_states=hidden)
    r.backend=SimpleNamespace(replay_session=contextlib.nullcontext,replay=replay)
    def load_batch(fb,pp):r.raw_num_token=fb.batch_size*8;r.bs=8;r._replay_graph_key='fixture'
    r.load_batch=load_batch
    worker_path=C/'srt/speculative/dflash_worker_v2.py'
    wm=extract(worker_path,['_aux_output_owners','_append_target_hidden_to_draft_kv_by_loc','forward_batch_generation'],{'nullcontext':contextlib.nullcontext},'DFlashWorkerV2')
    worker=SimpleNamespace(target_worker=SimpleNamespace(model_runner=SimpleNamespace(prefill_cuda_graph_runner=None,decode_cuda_graph_runner=r)))
    for name,fn in wm.items():setattr(worker,name,MethodType(fn,worker))
    reads=[];worker._append_target_hidden_to_draft_kv_by_loc_impl=lambda **kwargs:reads.append(kwargs['target_hidden'].clone())
    for generation,bs in enumerate([1,8,3,1,8]):
        inputs.copy_(torch.arange(64*H,device=a.device).reshape(64,H).remainder(31)/8+generation)
        fb=SimpleNamespace(batch_size=bs,forward_mode=mode.TARGET_VERIFY)
        out=r.execute(fb)
        check('actual_execute_ready_%d'%generation,lease._state=='ready' and out.hidden_states.shape==(bs*8,H*5) and out.marker=='preserved')
        worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=out.hidden_states)
        ref=old(model,None,None,staticfb,inputs)[1]
        check('actual_execute_values_%d'%generation,torch.equal(reads[-1],torch.cat(ref,-1)[:bs*8]) and lease._state=='idle')
    # Actual runner prevents replay after canonical storage replacement.
    original=buffers.aux_hidden_states;buffers.aux_hidden_states=original.clone()
    try:r.execute(SimpleNamespace(batch_size=1,forward_mode=mode.TARGET_VERIFY))
    except RuntimeError:check('actual_execute_replaced_pointer_rejected',lease._state=='idle')
    else:check('actual_execute_replaced_pointer_rejected',False)
    buffers.aux_hidden_states=original
    # Outer worker abandons only this invocation's ready output on pre-consume error.
    def fail(*args,**kwargs):r.execute(SimpleNamespace(batch_size=1,forward_mode=mode.TARGET_VERIFY));raise ValueError('before consumer')
    worker._forward_batch_generation_with_aux_output=fail
    try:worker.forward_batch_generation(None)
    except ValueError:pass
    check('decode_worker_exception_recovery',lease._state=='idle')
    out=r.execute(SimpleNamespace(batch_size=1,forward_mode=mode.TARGET_VERIFY));worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=out.hidden_states)
    check('decode_after_exception_reuses_buffer',lease._state=='idle')
    for reason,fb in [('mode',SimpleNamespace(batch_size=1,forward_mode=mode.DECODE)),('empty',SimpleNamespace(batch_size=0,forward_mode=mode.TARGET_VERIFY))]:
        try:r.execute(fb)
        except RuntimeError:check('execute_reject_'+reason,lease._state=='idle')
        else:check('execute_reject_'+reason,False)
    out=r.execute(SimpleNamespace(batch_size=1,forward_mode=mode.TARGET_VERIFY));generation=lease.generation
    try:r.execute(SimpleNamespace(batch_size=1,forward_mode=mode.TARGET_VERIFY))
    except RuntimeError:check('pending_generation_blocks_replay',lease.generation==generation and lease._state=='ready')
    else:check('pending_generation_blocks_replay',False)
    worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=out.hidden_states)
    prefill=baseowner.PrefillAuxOutputOwner(64,H,spec[1],a.device)
    worker.target_worker.model_runner.prefill_cuda_graph_runner=SimpleNamespace(prefill_aux_output_owner=prefill)
    token=prefill.begin_produce();prefill.end_produce(token,prefill.buffer)
    out=r.execute(SimpleNamespace(batch_size=1,forward_mode=mode.TARGET_VERIFY));worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=out.hidden_states)
    check('consumer_selects_decode_only',lease._state=='idle' and prefill._state=='ready')
    worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=prefill.buffer)
    check('consumer_selects_prefill_only',lease._state=='idle' and prefill._state=='idle')

if a.device!='cpu':torch.cuda.synchronize()
a.out.write_text(json.dumps({'status':'PASS','device':a.device,'checks':checks,'check_count':len(checks),'scope':'Exact source allocation/sharing/capture binding/replay trim/worker wrappers; registry and attention/model math fixtures, not full runtime.'},indent=2)+'\n');print(json.dumps({'status':'PASS','checks':len(checks)}))
