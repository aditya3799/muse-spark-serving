"""Independent CPU-only release source contracts; NEVER imports torch/CUDA.

AST extraction and metadata/tensor fakes do not validate native kernels, H200,
judge binary, distributed execution or output numerics. Missing assembly is HELD.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace as N
P=Path(__file__).resolve().parent;ROOT=P.parent
REL=Path('python/sglang/kernels/ops/moe/muse_spark_humming_split.py')
MODEL_REL=Path('python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py')
SCORED=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'
FROZEN=ROOT/'swarm_2026-11-17/deep_research/expert_partition/candidate_export'
parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,default=P/'candidate_sources')
parser.add_argument('--output',type=Path,default=P/'BUILD_CONTRACTS_CPU_RESULT.json');args=parser.parse_args()
report={'status':'RUNNING_CPU_ONLY','checks':[],'negative_controls':[],
        'scope':'AST/source contracts and CPU fakes only. No native GPU/judge/H200 evidence.'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return ast.parse(p.read_text(encoding='utf-8-sig'))
def functions(t,name):return [n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name]
def function(t,name,last=False):
    nodes=functions(t,name);assert nodes,'missing '+name
    return copy.deepcopy(nodes[-1 if last else 0])
def dump(n):return ast.dump(n,include_attributes=False)
def check(name,ok,detail=None):
    if not ok:raise AssertionError(name)
    report['checks'].append({'name':name,'detail':detail})

class StripImports(ast.NodeTransformer):
    def visit_Import(self,node):return None
    def visit_ImportFrom(self,node):return None

def compile_function(t,name,env,last=False):
    node=function(t,name,last);node.decorator_list=[];node=StripImports().visit(node)
    ast.fix_missing_locations(node)
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),'<extracted '+name+'>','exec'),env)
    return env[name]

class Tensor:
    def __init__(self,shape,dtype='bf16',tag='',cuda=True,device=0,contiguous=True,storage=None):
        self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype;self.tag=tag
        self.is_cuda=cuda;self.device=device;self.contiguous_=contiguous;self.storage=object() if storage is None else storage
    def numel(self):return math.prod(self.shape)
    def is_contiguous(self):return self.contiguous_
    def view(self,*shape):return Tensor(shape,self.dtype,self.tag,self.is_cuda,self.device,storage=self.storage)
    def flatten(self):return self.view(self.numel())
    def __getitem__(self,s):
        assert isinstance(s,slice)
        return self.view(len(range(*s.indices(self.numel()))))

TORCH=N(bfloat16='bf16',int32='i32',int64='i64')
TORCH.empty=lambda shape,**kw:Tensor(shape,kw.get('dtype','bf16'),device=kw.get('device',0))

def layer():
    return N(w13_post_scale=Tensor((32,),tag='w13_scale'),w2_post_scale=Tensor((32,),tag='w2_scale'),
             humming_metas={'w2':N(shape_k=4096)})

def run_preservation(scored,candidate,frozen):
    check('scored_small_function_ast_exact',dump(function(scored,'split_expert_outputs'))==dump(function(candidate,'split_expert_outputs')))
    check('scored_selector_ast_exact',dump(function(scored,'choose_partition_configs'))==dump(function(candidate,'choose_partition_configs')))
    check('scored_partition_ast_exact',dump(function(scored,'partition_routes'))==dump(function(candidate,'partition_routes')))
    check('public_wrapper_last',len(functions(candidate,'split_expert_outputs'))==2)
    alias=[n for n in candidate.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_original_split_expert_outputs' for x in n.targets)]
    check('frozen_original_alias_preserved',len(alias)==1 and isinstance(alias[0].value,ast.Name) and alias[0].value.id=='split_expert_outputs')
    original=function(candidate,'split_expert_outputs');wrapper=function(candidate,'split_expert_outputs',True)
    check('original_alias_before_wrapper',original.lineno<alias[0].lineno<wrapper.lineno)
    for name in ('_split_large_prefill_expert_outputs','partition_routes_scalable','_partial_hist','_reduce_hist','_classify'):
        check('frozen_export_ast_'+name,dump(function(candidate,name))==dump(function(frozen,name)))


def run_wrapper(candidate):
    calls=[]
    def spy(name):
        def invoke(*a,**kw):calls.append((name,a,kw));return name
        return invoke
    env={'_original_split_expert_outputs':spy('scored'),'_split_large_prefill_expert_outputs':spy('large')}
    wrapper=compile_function(candidate,'split_expert_outputs',env,True)
    original_helper=object();x=Tensor((768,4096))
    for enabled in (False,True,1,None,'true'):
        for routes in (0,8,4096,4104,6144,131072,131080):
            for retain in (False,True):
                for defer in (False,True):
                    obj=layer();obj._humming_large_prefill_partition_enabled=enabled
                    ids=Tensor((routes//8,8),'i32');calls.clear()
                    result=wrapper(original_helper,obj,x,ids,TORCH,256,retain=retain,defer_downscale=defer)
                    expected='large' if enabled is True and routes>4096 else 'scored'
                    assert result==expected and len(calls)==1
                    _,a,kw=calls[0]
                    assert a==(original_helper,obj,x,ids,TORCH,256) and kw==dict(retain=retain,defer_downscale=defer)
    check('wrapper_all_flags_identity_forwarding',True,'140 combinations; too-large dispatch is rejected inside frozen large helper')
    obj=layer();ids=Tensor((768,8),'i32');calls.clear()
    assert wrapper(original_helper,obj,x,ids,TORCH,256)=='scored'
    check('wrapper_missing_enable_defaults_scored',True)


def run_helper_trace(candidate):
    for retain in (False,True):
        for defer in (False,True):
            events=[];fallback_calls=[]
            def fallback(*a,**kw):fallback_calls.append((a,kw));return 'fallback'
            configs={stage:{'block_shape':[16,128,64],'warp_shape':[16,32,64],
                            'use_f16_accum':False,'use_stream_k':False,'tag':part}
                     for part in () for stage in ()}
            plan={part:{stage:{'block_shape':[16,128,64],'tag':part} for stage in ('w13','w2')} for part in ('light','heavy')}
            plan['cutoff']=16
            def partition(ids,e,c):
                events.append(('partition',e,c));return Tensor(ids.shape,'i32','light'),Tensor(ids.shape,'i32','heavy'),Tensor((e,),'i32')
            def align(ids,block,experts,**kw):
                events.append(('align',ids.tag,block,experts));return ids,object(),object()
            def forward(l,**kw):events.append(('gemm',kw['sublayer_name'],kw['tuning_config']['tag'],kw['outputs'].storage))
            def scale(out,ids,s):events.append(('scale',s.tag))
            def activation(a,b,**kw):events.append(('activation',))
            env={'HummingMethod':N(forward_layer=forward),'moe_align_block_size':align,
                 'silu_and_mul':activation,'scale_active_rows':scale,
                 '_choose_large_prefill_configs':lambda *a:copy.deepcopy(plan),'partition_routes_scalable':partition}
            helper=compile_function(candidate,'_split_large_prefill_expert_outputs',env)
            obj=layer();x=Tensor((768,4096));ids=Tensor((768,8),'i32')
            output=helper(fallback,obj,x,ids,TORCH,256,retain=retain,defer_downscale=defer)
            assert not fallback_calls and output[0].shape==(768,8,4096) and output[1] is ids
            math_events=[e[:3] for e in events if e[0] in ('gemm','scale','activation')]
            expected=[('gemm','w13','light'),('gemm','w13','heavy'),('scale','w13_scale'),('activation',),
                      ('gemm','w2','light'),('gemm','w2','heavy')]
            if not defer:expected.append(('scale','w2_scale'))
            assert math_events==expected
            gemms=[e for e in events if e[0]=='gemm']
            assert all(e[3] is gemms[0][3] for e in gemms),'up/down alias boundary changed'
            check('both_up_before_scale_activation_down',True,dict(retain=retain,defer_downscale=defer))
            for m in (0,16385):
                fallback_calls.clear();events.clear();badids=Tensor((m,8),'i32');badx=Tensor((m,4096))
                assert helper(fallback,obj,badx,badids,TORCH,256,retain=retain,defer_downscale=defer)=='fallback'
                assert fallback_calls==[((obj,badx,badids,TORCH),dict(retain=retain,defer_downscale=defer))] and not events
            check('empty_too_large_reject_without_launch',True,dict(retain=retain,defer_downscale=defer))
            env['_choose_large_prefill_configs']=lambda *a:None
            fallback_calls.clear();events.clear()
            assert helper(fallback,obj,x,ids,TORCH,256,retain=retain,defer_downscale=defer)=='fallback'
            assert not events
            check('selector_rejection_no_partial_launch',True)


def run_setup_order(base_model,new_model):
    original=function(base_model,'_prepare_humming_experts');new=function(new_model,'_prepare_humming_experts')
    calls=[n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='prepare_large_prefill_partition']
    check('one_prepare_call',len(calls)==1)
    owners=[n for n in ast.walk(new) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Attribute) and t.value.attr=='fp32_configs' for t in n.targets)]
    check('prepare_after_owned_fp32_configs',bool(owners) and calls[0].lineno>max(n.lineno for n in owners))
    # Remove ONLY new helper import/call and demand original setup semantics.
    class RemoveHook(ast.NodeTransformer):
        def visit_ImportFrom(self,n):
            return None if any(x.name=='prepare_large_prefill_partition' for x in n.names) else n
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='prepare_large_prefill_partition':return None
            return n
    cleaned=RemoveHook().visit(copy.deepcopy(new))
    check('setup_only_adds_prepare_hook',dump(cleaned)==dump(original))
    # Execute actual config construction tail with setup spy and authentic shape.
    start=next(i for i,n in enumerate(new.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='fp32_configs' for t in n.targets))
    tail=copy.deepcopy(new);tail.name='setup_tail';tail.args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='layer')],kwonlyargs=[],kw_defaults=[],defaults=[])
    tail.body=StripImports().visit(ast.Module(body=tail.body[start:],type_ignores=[])).body
    defaults={stage:[(0,131072,{'use_stream_k':True,'use_f16_accum':True})] for stage in ('w13','w2')};observed=[]
    def prepare(l,t):
        assert set(l.fp32_configs)=={'w13','w2'}
        for stage in defaults:
            assert l.fp32_configs[stage] is not defaults[stage]
            assert l.fp32_configs[stage][0][2]['use_stream_k'] is False
            assert l.fp32_configs[stage][0][2]['use_f16_accum'] is False
            assert defaults[stage][0][2]=={'use_stream_k':True,'use_f16_accum':True}
        observed.append(l)
    env={'copy':copy,'torch':TORCH,'prepare_large_prefill_partition':prepare,
         'HummingMethod':N(get_default_tuning_configs=lambda l,**kw:defaults[kw['sublayer_name']])}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tail],type_ignores=[])),'<setup_tail>','exec'),env)
    l=N();assert env['setup_tail'](l) is l and observed==[l]
    check('actual_setup_tail_owned_tables_before_enable',True)


def run_admission_selector(candidate):
    """Execute assembled signature/preparer/selector; native validators are spies.

    Hopper's separate native-validator CPU suite covers validator semantics.
    This test covers their INTEGRATION and fail-closed flow, not SM90 legality.
    """
    from dataclasses import dataclass,field,fields
    from contextlib import nullcontext
    @dataclass
    class LayerConfig:
        shape_n:int=256
        shape_k:int=4096
        def __post_init__(self):pass
    @dataclass
    class ComputeConfig:
        gemm_type:str='indexed'
        use_f16_accum:bool=False
        def __post_init__(self):pass
    @dataclass
    class TuningConfig:
        block_shape:object=None
        warp_shape:object=None
        use_stream_k:bool=False
        num_ctas_per_sm:int=1
        num_write_splits:int=1
        def __post_init__(self):pass
    validations=[]
    class HummingKernel:
        def __init__(self,**kw):
            self.__dict__.update(kw)
            self.num_threads=32*math.prod(b//w for b,w in zip(self.block_shape,self.warp_shape))
            self.__post_init__()
        def check_shape(self):validations.append('shape')
        def check_dtype(self):validations.append('dtype')
        def check_scale(self):validations.append('scale')
        def check_config(self):validations.append('config')
        def select_mma_op_class(self):validations.append('mma');return object
    smem={'fits':True}
    env={'deepcopy':copy.deepcopy,'fields':fields,'LayerConfig':LayerConfig,
         'ComputeConfig':ComputeConfig,'TuningConfig':TuningConfig,'HummingKernel':HummingKernel,
         'fits_device_smem':lambda *a:smem['fits']}
    signature=compile_function(candidate,'_large_prefill_config_signature',env)
    prepare=compile_function(candidate,'prepare_large_prefill_partition',env)
    compile_function(candidate,'select_ep_config',env)
    original=compile_function(candidate,'choose_partition_configs',env)
    env['original_choose_partition_configs']=original
    choose=compile_function(candidate,'_choose_large_prefill_configs',env)
    cuda=N(device=lambda dev:nullcontext(),get_device_properties=lambda dev:N(major=9,minor=0,max_threads_per_block=1024))
    fake_torch=N(device=lambda value:N(type='cuda'),cuda=cuda)
    light=dict(block_shape=[16,128,64],warp_shape=[16,32,32],use_f16_accum=False,use_stream_k=False)
    heavy=dict(block_shape=[64,128,128],warp_shape=[16,32,64],use_f16_accum=False,use_stream_k=False)
    def make_layer():
        x=layer();x.humming_metas={stage:N(shape_n=8192 if stage=='w13' else 4096,shape_k=4096,
            num_experts=32,a_dtype='bfloat16',c_dtype='bfloat16',b_dtype='float4e2m1',bs_dtype='float8e4m3',
            input_scale_group_size=0,weight_scale_group_size=16,has_zero_point=False,has_bias=False,
            use_packed_k_layout=False) for stage in ('w13','w2')}
        x.fp32_configs={stage:[(0,1024,copy.deepcopy(light)),(1024,131072,copy.deepcopy(heavy))] for stage in ('w13','w2')}
        return x
    obj=make_layer();before=copy.deepcopy(obj.fp32_configs)
    assert prepare(obj,fake_torch) is True and obj._humming_large_prefill_partition_enabled is True
    assert type(obj._humming_large_prefill_configs) is frozenset and validations
    assert obj.fp32_configs==before
    check('assembled_prepare_records_immutable_admission_without_table_mutation',True,'Native validator methods are CPU spies; no kernel preparation/load/launch')
    plan=choose(obj,6144,256)
    assert plan is not None
    for stage in ('w13','w2'):
        assert plan['light'][stage]['block_shape'][2]==plan['heavy'][stage]['block_shape'][2]==128
        assert plan['light'][stage]['warp_shape'][2]==plan['heavy'][stage]['warp_shape'][2]==64
        assert all((stage,signature(plan[kind][stage])) in obj._humming_large_prefill_configs for kind in ('light','heavy'))
    assert obj.fp32_configs==before
    check('assembled_selector_uses_actual_prepared_signatures_and_preserves_K',True)
    tuple_config=copy.deepcopy(light);tuple_config['block_shape']=tuple(tuple_config['block_shape']);tuple_config['warp_shape']=tuple(tuple_config['warp_shape'])
    check('signature_shape_list_tuple_identity',signature(light)==signature(tuple_config))
    for name,mutate in [
        ('missing_admission',lambda x:setattr(x,'_humming_large_prefill_configs',frozenset())),
        ('mutable_admission',lambda x:setattr(x,'_humming_large_prefill_configs',set(x._humming_large_prefill_configs))),
        ('changed_stage_config',lambda x:x.fp32_configs['w13'][1][2].update(warp_shape=[16,32,32])),
        ('malformed_interval',lambda x:x.fp32_configs['w13'].append((1000,7000,copy.deepcopy(heavy)))),
        ('wrong_scale_count',lambda x:setattr(x,'w2_post_scale',Tensor((31,))))]:
        x=make_layer();assert prepare(x,fake_torch)
        mutate(x);assert choose(x,6144,256) is None
        report['negative_controls'].append(name)
    assert choose(obj,131073,256) is None
    assert choose(obj,6144,255) is None
    report['negative_controls']+=['too_large_selector','nondivisible_global_experts']
    # Native capability rejection clears any stale enable/admission.
    smem['fits']=False
    assert prepare(obj,fake_torch) is False
    assert obj._humming_large_prefill_partition_enabled is False and obj._humming_large_prefill_configs==frozenset()
    check('setup_native_rejection_clears_previous_enable',True)
    # Absent imports/changed APIs fail closed, rather than enabling generic work.
    broken_env={'deepcopy':copy.deepcopy,'fields':fields}
    missing=compile_function(candidate,'prepare_large_prefill_partition',broken_env)
    obj._humming_large_prefill_partition_enabled=True
    assert missing(obj,fake_torch) is False and obj._humming_large_prefill_partition_enabled is False
    check('setup_missing_api_falls_back',True)

def main():
    paths=[SCORED/REL,FROZEN/REL,args.candidate/REL,SCORED/MODEL_REL,args.candidate/MODEL_REL]
    if any(not p.exists() for p in paths):
        report.update(status='HELD_CANDIDATE_NOT_ASSEMBLED',missing=[str(p) for p in paths if not p.exists()]);return
    base,candidate,frozen=tree(paths[0]),tree(paths[2]),tree(paths[1])
    run_preservation(base,candidate,frozen);run_wrapper(candidate);run_helper_trace(candidate)
    run_setup_order(tree(paths[3]),tree(paths[4]))
    run_admission_selector(candidate)
    report['source_sha256']={str(p):sha(p) for p in paths}
    report['status']='PASS_CPU_BUILD_CONTRACTS_NATIVE_PENDING'
try:
    main()
except BaseException as error:
    report.update(status='FAIL_CPU_BUILD_CONTRACT',error=repr(error))
    args.output.write_text(json.dumps(report,indent=2)+'\n');raise
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(report['checks'])}))


