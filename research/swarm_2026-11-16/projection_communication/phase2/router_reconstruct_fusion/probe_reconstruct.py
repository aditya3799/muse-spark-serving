"""Exact split2 reconstruction gate; original full GEMM geometry is preserved."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, os, statistics, sys, time
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert os.environ.get('RECONSTRUCT_BOUND')=='360'
a.out.mkdir(parents=True,exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((P/'manifest.json').read_text())
for rel,hashes in manifest['sources'].items():
    assert sha(a.workspace/'sglang'/rel)==hashes['baseline']
    assert sha(P/'candidate_sources'/rel)==hashes['candidate']
sys.path.insert(0,str(a.workspace/'optimization_marlin_tiles_2026-11-15'))
from kernel_runtime import bootstrap
provenance=bootstrap(a.workspace/'sglang')
import torch
from sglang.kernels.ops.moe.muse_spark_routing_jit import fused_muse_spark_routing_jit as stock
rel='python/sglang/kernels/ops/moe/muse_spark_routing_jit.py'
spec=importlib.util.spec_from_file_location('candidate_reconstruct',P/'candidate_sources'/rel)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
# Only source path routing differs from installation; no kernel/arguments altered.
original_load=mod.load_jit
header=P/'candidate_sources/python/sglang/kernels/jit/csrc/muse_spark/routing.cuh'
def local_load(*args,**kwargs):
    assert kwargs['cuda_files']==['muse_spark/routing.cuh']
    kwargs['cuda_files']=[str(header)]
    return original_load(*args,**kwargs)
mod.load_jit=local_load
candidate=mod.fused_muse_spark_routing_jit
report={'status':'running','scope':'Single RTX3050 component, exact pinned full-GEMM shape and router. No serving, H200, public fused-op dispatch or collectives.',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'sources':manifest['sources'],'checks':[],'timings':[],
        'harness_sha256':sha(Path(__file__)),'fixed_gates':{'ids':'bitwise','weights':'bitwise','packed':'bitwise','control_deviation':.1,'samples':12}}
def save():
    temp=a.out/'report.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(a.out/'report.json')
def packed(m,k):return torch.empty((m,k),device='cuda',dtype=torch.int32)
def invoke(fn,x,bias,**kw):
    return fn(x,bias,8,1.,router_output_multiplier=.17677669529663687,
              router_score_division_eps=1e-15,packed_topk_ids=packed(x.shape[0],8),**kw)
def old(acc,bias,**kw):
    hi,lo=acc.chunk(2,dim=-1)
    return invoke(stock,hi+lo,bias,**kw)
def new(acc,bias,**kw):return invoke(candidate,acc,bias,sum_halves=True,**kw)
def check(label,aa,bb,valid=None,**ctx):
    mismatches=[]
    for i,(x,y) in enumerate(zip(aa,bb)):
        # Stock leaves padded-row weights uninitialized; IDs/packed are defined.
        if i==0 and valid is not None:x=x[:valid];y=y[:valid]
        if x.dtype==torch.float32:x=x.view(torch.int32);y=y.view(torch.int32)
        mismatches.append(int((x!=y).sum().item()))
    row={'label':label,'context':ctx,'mismatches_weights_ids_packed':mismatches,'passed':not any(mismatches)}
    report['checks'].append(row);save()
    if not row['passed']:
        report.update(status='numerical_failure',failure=row);save();raise AssertionError(row)
def capture(fn,repeat=1):
    for _ in range(3):fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeat):out=fn()
    return g,out
save();torch.set_grad_enabled(False);torch.manual_seed(900)
# Direct input edge gate includes exact cancellation, signed zeros and subnormals.
for m in (1,8,64,512,4096):
    bias=torch.linspace(-.01,.01,256,device='cuda',dtype=torch.float32)
    acc=torch.randn((m,512),device='cuda',dtype=torch.float32)
    for pattern in ('random','signed_zero','subnormal','cancellation','near_ties'):
        if pattern=='random':acc.normal_()
        elif pattern=='signed_zero':acc.zero_();acc[:,1::2]=-0.
        elif pattern=='subnormal':acc.copy_(torch.arange(512,device='cuda',dtype=torch.int32).view(torch.float32).expand(m,-1))
        elif pattern=='cancellation':acc[:,:256].normal_();acc[:,256:].copy_(-acc[:,:256])
        else:
            acc.zero_();acc[:,256:]=torch.linspace(-1e-7,1e-7,256,device='cuda');bias.zero_()
        check('direct_'+pattern,old(acc,bias),new(acc,bias),m=m)
    view=torch.randn((m,1024),device='cuda',dtype=torch.float32)[:,::2]
    check('noncontiguous_fallback',old(view,bias),new(view,bias),m=m)
    # Optional local expert remap plus padded rows preserve the stock sentinels.
    mapping=torch.full((256,),-1,device='cuda',dtype=torch.int32);mapping[64:96]=torch.arange(32,device='cuda',dtype=torch.int32)
    valid=torch.tensor([max(0,m-2)],device='cuda',dtype=torch.int32)
    kw={'local_expert_mapping':mapping,'num_valid_tokens':valid}
    check('mapping_padding',old(acc,bias,**kw),new(acc,bias,**kw),valid=max(0,m-2),m=m)
    ga,oa=capture(lambda:old(acc,bias));gb,ob=capture(lambda:new(acc,bias))
    for seed in (910,911,912):
        torch.manual_seed(seed);acc.normal_();bias.normal_(0,.001);ga.replay();gb.replay()
        check('changed_input_graph',oa,ob,m=m,seed=seed)
    del ga,gb,oa,ob
    print(json.dumps({'milestone':'direct_shape_done','m':m}),flush=True)
# Actual pinned router's full [M,8192] @ [8192,512] output is unchanged.
model=a.workspace/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/srt/models/muse_spark.py'
assert sha(model)=='255b2ce03ac0ecc500e230c74b8f3966144bf7e8009b99bd77d77dd9a470f0c6'
node=next(n for n in ast.parse(model.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xRouter')
ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional};exec(compile(ast.Module(body=[node],type_ignores=[]),str(model),'exec'),ns)
router=ns[node.name](8192,256,torch.float32).cuda();router.weight.normal_(0,1/90)
for m in (8,64,512,4096):
    x=torch.randn((m,8192),device='cuda',dtype=torch.bfloat16)*.05
    for pattern in ('random','near_ties'):
        router.weight.normal_(0,1/90)
        if pattern=='near_ties':router.weight[1::2].copy_(router.weight[::2]);router.weight[1::2,0].add_(1e-7)
        router._build_split2()
        acc=torch.mm(x.to(torch.bfloat16),router._split2_weight,out_dtype=torch.float32)
        logits=router.forward_split2(x)
        assert torch.equal(logits.view(torch.int32),(acc[:,:256]+acc[:,256:]).view(torch.int32))
        check('same_full_gemm_'+pattern,invoke(stock,logits,router.e_score_correction_bias),new(acc,router.e_score_correction_bias),m=m)
    # Same matmul/weight/input, fused reconstruction differs only after GEMM.
    def full_a():return invoke(stock,router.forward_split2(x),router.e_score_correction_bias)
    def full_b():return new(torch.mm(x.to(torch.bfloat16),router._split2_weight,out_dtype=torch.float32),router.e_score_correction_bias)
    for scope,fa,fb,repeat in (('reconstruct_and_route',lambda:old(acc,router.e_score_correction_bias),lambda:new(acc,router.e_score_correction_bias),16),('full_router',full_a,full_b,1)):
        ga,oa=capture(fa,repeat);gb,ob=capture(fb,repeat);check('timing_graph_output',oa,ob,m=m,scope=scope)
        def sample(g):
            st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();g.replay();en.record();en.synchronize();return st.elapsed_time(en)*1000/repeat
        samples={k:[] for k in ('A1','B1','A2','B2')}
        for i in range(12):
            for label in (list(samples) if i%2==0 else list(reversed(samples))):samples[label].append(sample(ga if label[0]=='A' else gb))
        med={k:statistics.median(v) for k,v in samples.items()};arms={k:statistics.median(samples[k+'1']+samples[k+'2']) for k in 'AB'};control={k:med[k+'1']/med[k+'2'] for k in 'AB'}
        report['timings'].append({'m':m,'scope':scope,'samples_us':samples,'arm_medians_us':arms,'speedup':arms['A']/arms['B'],'controls':control,'noise_pass':all(abs(v-1)<=.1 for v in control.values()),'graph_body':repeat,'warm_residency':True})
        save();del ga,gb,oa,ob
    print(json.dumps({'milestone':'full_shape_done','m':m}),flush=True)
report.update(status='completed',passed=True);save();print(json.dumps({'status':'completed','checks':len(report['checks'])}),flush=True)
