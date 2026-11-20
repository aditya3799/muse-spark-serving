from pathlib import Path
import ast,types,json,sys,random,hashlib,typing,subprocess,os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];D=HERE/'integration'
class Comm:
 def __init__(self):
  self.disabled=False;self.has_multicast=True;self.world_size=8;self.max_push_size=2**24;self.device='cuda:0';self.config=types.SimpleNamespace(num_push_blocks=132)
module=types.ModuleType('sglang.srt.distributed.device_communicators.custom_all_reduce_v2');module.CustomAllReduceV2=Comm;sys.modules[module.__name__]=module
arch=[9,0];sm=[132];torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda:tuple(arch),get_device_properties=lambda d:types.SimpleNamespace(multi_processor_count=sm[0])))
ns={'torch':torch,'Optional':typing.Optional,'CustomAllReduceV2':Comm,'_VEC':8,'_THREAD_CHOICES':(1024,512,256,128),'_CLUSTER_SIZES':(1,2,4,8,16),'_SMALL_M':192,'_SMALL_M_THREAD_CHOICES':(256,512,1024,128)}
base=ROOT/'sglang/python/sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py';candidate=D/'python/sglang/kernels/ops/communication/muse_spark_all_gather_norm_residual.py'
functions=[n for n in ast.parse(base.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_launch_config','cluster_size_for_num_tokens','_cluster_num_threads','_cluster_size_for_shape','norm_residual_launch_config','_cluster_launch_config')]
functions +=[n for n in ast.parse(candidate.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_supports_multicast_norm','_gather_norm_launch_config','all_gather_norm_residual_supported')]
for f in functions:f.decorator_list=[]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(candidate),'exec'),ns)
checks=[]
for major in (8,9,10):
 arch[0]=major
 for m in (1,8,16,32,64,128,192,256,511,512):
  cfg=ns['_gather_norm_launch_config'](8192,m);support=ns['all_gather_norm_residual_supported'](Comm(),m,8192)
  assert support==(major>=9 and (cfg[0]==1 or m*cfg[0]<=132))
  if major==9:assert cfg==(1,256 if m<192 else 512,m<192)
  if major==10:assert cfg==ns['norm_residual_launch_config'](8192,m)
  checks.append({'architecture':major,'rows':m,'config':cfg,'support':support})
arch[0]=9
for field,value in [('disabled',True),('has_multicast',False),('world_size',1),('max_push_size',0)]:
 c=Comm();setattr(c,field,value);assert not ns['all_gather_norm_residual_supported'](c,32,8192)
c=Comm();c.config.num_push_blocks=1;assert not ns['all_gather_norm_residual_supported'](c,32,8192)
for m,h in [(0,8192),(-1,8192),(32,8191),(32,768)]:assert not ns['all_gather_norm_residual_supported'](Comm(),m,h)
c=Comm();c.config.num_push_blocks=256;assert not ns['all_gather_norm_residual_supported'](c,256,8192);assert ns['all_gather_norm_residual_supported'](c,8,8192)
traces=[];rng=random.Random(971314)
for cap in (20,114,132):
 counters=[0]*cap
 for iteration in range(256):
  m=rng.choice((1,8,32,128,192,256,512));kind='native' if iteration%3==0 else 'fused';phase=iteration%2
  grid=min((m*8192//8+127)//128,cap-1)+1 if kind=='native' else min(m,cap)
  assert all(c==phase for c in counters)
  owned=list(range(grid-1 if kind=='native' else grid));extra=list(range(grid-1 if kind=='native' else grid,cap))
  assert sorted(owned+extra)==list(range(cap));assert not(set(owned)&set(extra))
  for i in owned+extra:counters[i]=phase^1
  assert all(c==phase^1 for c in counters)
 traces.append({'counter_cap':cap,'alternating_operations':256})
model=D/'python/sglang/srt/models/muse_spark.py';tree=ast.parse(model.read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_reduce_and_unpool')
namespace={'_is_cpu':False,'get_forward':lambda:types.SimpleNamespace(sp_active=False),'get_parallel':lambda:types.SimpleNamespace(tp_group=types.SimpleNamespace(ca_comm=Comm())),'tensor_model_parallel_all_reduce':lambda x:x,'gemm_ag_supported':lambda *a:False,'small_projection':lambda *a:'ordinary_gather','small_projection_local':lambda *a:'deferred_local','bf16_gemm_dispatch':lambda *a:'dense'}
body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method];exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(model),'exec'),namespace)
self=types.SimpleNamespace(tp_size=8,moe_tp_size=1,post_expert_proj=types.SimpleNamespace(weight=object()),_can_shard_prefill_row_projection=lambda *a:False,_can_shard_small_projection=lambda *a:True,_defers_unpool_gather=lambda *a:True)
x=types.SimpleNamespace(shape=(32,4096));assert namespace['_reduce_and_unpool'](self,x)=='deferred_local'
self._defers_unpool_gather=lambda *a:False;assert namespace['_reduce_and_unpool'](self,x)=='ordinary_gather'
self._can_shard_small_projection=lambda *a:False;assert namespace['_reduce_and_unpool'](self,x)=='dense'
report={'passed':True,'dispatch_checks':checks,'negative_checks':10,'bounded_grid_positive_check':True,'model_branch_checks':3,'phase_traces':traces,'scope':'CPU source-extracted dispatch and counter index model, not distributed memory-order proof','candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()}
(D/'structural_checks.json').write_text(json.dumps(report,indent=2)+'\n')
# A private index composes the exact frozen release; never alter the user's index/worktree.
index=D/'compose.index';env=dict(os.environ,GIT_INDEX_FILE=str(index));git=['git','-C',str(ROOT/'sglang')]
def call(*args):return subprocess.check_output(git+list(args),env=env)
call('read-tree','297e54fd105e974cdf401e5b4d20253a22c16f90');call('apply','--cached',str(ROOT/'release_2026-11-18/candidate.diff'));call('apply','--cached',str(D/'delta.diff'));call('diff','--cached','--check')
patch=call('diff','--cached','--binary','297e54fd105e974cdf401e5b4d20253a22c16f90');paths=call('diff','--cached','--name-only','297e54fd105e974cdf401e5b4d20253a22c16f90').decode().splitlines()
assert all(n.startswith('python/sglang/') and not n.startswith('python/sglang/srt/model_loader/') for n in paths);assert len(patch)<=400000
maxline=max(len(x) for x in patch.splitlines() if x.startswith(b'+'));assert maxline<=400
(D/'candidate.diff').write_bytes(patch);index.unlink()
manifest=json.loads((D/'manifest.json').read_text());manifest.update(cumulative_sha256=hashlib.sha256(patch).hexdigest(),cumulative_bytes=len(patch),cumulative_paths=paths,max_added_line=maxline,structural_checks_passed=True)
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ('cumulative_sha256','cumulative_bytes','max_added_line','structural_checks_passed')}))
