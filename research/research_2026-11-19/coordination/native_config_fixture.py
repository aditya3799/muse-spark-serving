import argparse,ast,hashlib,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research_2026-11-18_target_moe'))
import bench_experts as fixture
for p in (fixture.OLD/'humming_vendor',fixture.OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15',ROOT/'research_2026-11-18_rl_prefill/gemm'):
 sys.path.insert(0,str(p))
from kernel_runtime import bootstrap
bootstrap(ROOT/'sglang')
from fixture_tuning_cache import cached_threshold_for_preparation
import torch
import native_config_search as search
parser=argparse.ArgumentParser();parser.add_argument('--rows',type=int,nargs='+',default=[8,32,128,256]);parser.add_argument('--seed',type=int,default=193781);args=parser.parse_args()
source=fixture.SOURCE/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
names={'_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs'}
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
moe=fixture.SOURCE/'kernels/ops/moe'
fixture.module('sglang.kernels.ops.moe.muse_spark_humming_scale',moe/'muse_spark_humming_scale.py')
fixture.module('sglang.kernels.ops.moe.muse_spark_humming_split',moe/'muse_spark_humming_split.py')
finish=fixture.module('policy_finalizer',moe/'muse_spark_post_expert_finalize_triton.py').fused_post_expert_norm_weighted_sum
report={'scope':'Synthetic original-format expert weights onlocalSM86; wholefrozenparentFFN andfinalizer; noH200servingclaim.','seed':args.seed,'parent_sha256':hashlib.sha256((fixture.RELEASE/'candidate.diff').read_bytes()).hexdigest(),'search_sha256':hashlib.sha256((HERE/'native_config_search.py').read_bytes()).hexdigest(),'device':torch.cuda.get_device_name(),'rows':{},'complete':False}
output=HERE/'native_config_controlled.json'
def save():output.write_text(json.dumps(report,indent=2))
save();torch.manual_seed(args.seed);raw=torch.nn.Module();h=4096;e=32
for stage,n in (('w13',2*h),('w2',h)):
 for name,t in (('weight',torch.randint(0,256,(e,n,h//2),device='cuda',dtype=torch.uint8)),('weight_scale',torch.randint(40,68,(e,n,h//16),device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn)),('weight_scale_2',torch.linspace(.125,.5,e,device='cuda',dtype=torch.bfloat16))):
  setattr(raw,stage+'_'+name,torch.nn.Parameter(t,requires_grad=False))
with cached_threshold_for_preparation(HERE):layer=ns['_prepare_humming_experts'](raw,torch)
del raw,t
norm=torch.linspace(.7,1.2,h,device='cuda',dtype=torch.bfloat16)
for m in args.rows:
 if m*8>4096:raise ValueError('firstpolicysearch coverssmall-route split only')
 x=torch.empty((m,h),device='cuda',dtype=torch.bfloat16);ids=torch.empty((m,8),device='cuda',dtype=torch.int32);scores=torch.empty((m,8),device='cuda')
 def update(seed,profile):
  g=torch.Generator().manual_seed(seed);weights=torch.rand(m,256,generator=g)
  if profile=='clustered':weights[::2,:8]+=2
  if profile=='all_local':weights[:,:32]+=2
  if profile=='all_remote':weights[:,32:]+=2
  routes=weights.topk(8).indices;ids.copy_(torch.where(routes<32,routes,-1))
  torch.manual_seed(seed);x.normal_(std=.023);scores.uniform_();scores.div_(scores.sum(1,keepdim=True))
 def run(view):
  y,loc=ns['_humming_expert_outputs'](view,x,ids,torch,defer_downscale=True,global_experts=256)
  return finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,expert_output_scale=view.w2_post_scale)
 update(args.seed,'uniform')
 report['rows'][str(m)]=search.search(layer,run,update,args.seed+m);save()
 print(m,report['rows'][str(m)]['chosen'],report['rows'][str(m)]['heldout_gain'],flush=True)
 torch.cuda.empty_cache()
report['complete']=True;report['peak_allocated']=torch.cuda.max_memory_allocated();save()
