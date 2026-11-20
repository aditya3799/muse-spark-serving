"""Windows CPU-only admission test: real pinned Humming validators, fake hardware.
No Torch installation, CUDA context, compiler or native module is used.
"""
import ast,contextlib,dataclasses,hashlib,importlib.util,json,math,sys,types,copy
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent;V=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming';N=types.SimpleNamespace
calls={'forbidden':0,'props':0};props=N(major=9,minor=0,max_threads_per_block=1024,shared_memory_per_block_optin=227*1024,shared_memory_per_block=48*1024,shared_memory_per_multiprocessor=228*1024)
def forbidden(*a,**k):calls['forbidden']+=1;raise AssertionError('runtime/compiler operation forbidden')
def package(name,path=None):
 m=types.ModuleType(name);m.__path__=[] if path is None else [str(path)];sys.modules[name]=m;return m
def module(name,**kw):
 m=types.ModuleType(name);m.__dict__.update(kw);sys.modules[name]=m;return m
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
class Device:
 def __init__(self,value='cuda:0'):
  text=str(value);self.type=text.split(':')[0];self.index=int(text.split(':')[1]) if ':' in text else None
 def __str__(self):return self.type+(':'+str(self.index) if self.index is not None else '')
 def __eq__(self,other):return str(self)==str(other)
def getprops(*a):calls['props']+=1;return props
torch=module('torch',device=Device,Tensor=type('Tensor',(),{}),dtype=type('dtype',(),{}))
for name in ('float8_e8m0fnu','float8_e4m3fn','float8_e5m2','float16','bfloat16','float32'):setattr(torch,name,'torch.'+name)
torch.cuda=N(device=lambda dev:contextlib.nullcontext(),get_device_properties=getprops,get_device_capability=lambda *a:(props.major,props.minor),current_device=lambda:0,set_device=forbidden)
package('humming',V);package('humming.utils',V/'utils');package('humming.jit',V/'jit');package('humming.kernel',V/'kernel')
package('cuda');package('cuda.bindings');module('cuda.bindings.driver',cuCtxGetCurrent=forbidden,cuModuleLoad=forbidden)
class Compiler:
 compile=staticmethod(forbidden)
module('humming.jit.compiler',NVCCCompiler=Compiler,NVRTCCompiler=Compiler)
module('humming.utils.cubin',get_cubin_kernel_names=forbidden)
module('humming.tune',get_heuristics_config=forbidden)
module('pynvml')
jit=module('humming.utils.jit');tree=ast.parse((V/'utils/jit.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='is_power_of_two');exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_is_power_of_two','exec'),jit.__dict__)
from humming.config import LayerConfig,ComputeConfig,TuningConfig
from humming.kernel.humming import HummingKernel
from humming.utils.device import _get_device_smem_limits
ns={};exec(compile((P/'large_partition_admission.py.txt').read_text(encoding='utf-8-sig'),'admission','exec'),ns);prepare=ns['prepare_large_prefill_partition'];signature=ns['_large_prefill_config_signature']
manifest=json.loads((ROOT/'swarm_2026-11-17/deep_research/coordination/SM90_K_PRESERVING_POLICIES.json').read_text());checks=[]
def layer():
 metas={};configs={}
 for stage in ('w13','w2'):
  rows=[r for r in manifest['rows'] if r['stage']==stage];full=rows[0]['configuration']
  metas[stage]=LayerConfig(**{f.name:full[f.name] for f in dataclasses.fields(LayerConfig) if f.init and f.name in full})
  unique={}
  for r in rows:
   for key in ('original_light','baseline'):unique[signature(r[key])]=copy.deepcopy(r[key])
  configs[stage]=[(i,i+1,c) for i,c in enumerate(unique.values())]
 return N(humming_metas=metas,fp32_configs=configs,w13_post_scale=N(device=Device()),w2_post_scale=N(device=Device()))
def check(name,condition):assert condition,name;checks.append(name)
b=layer();before=copy.deepcopy(b.fp32_configs);check('native SM90 finite policy set admitted',prepare(b,torch));check('tuning dictionaries unchanged',b.fp32_configs==before);check('immutable full stage signatures',type(b._humming_large_prefill_configs) is frozenset)
for r in manifest['rows']:
 check('native repaired '+r['stage']+' '+str(r['first_token_rows']),(r['stage'],signature(r['corrected_light'])) in b._humming_large_prefill_configs)
check('no runtime/compiler/context operation',calls['forbidden']==0)
check('signature list tuple equivalence',signature({'x':[1,2]})==signature({'x':(1,2)}));check('signature bool int distinguished',signature({'x':True})!=signature({'x':1}));check('full launcher field fingerprint',signature({'num_sms':132})!=signature({'num_sms':131}))
for name,modify in [
 ('zero block dimension',lambda c:c['block_shape'].__setitem__(0,0)),
 ('negative warp dimension',lambda c:c['warp_shape'].__setitem__(2,-1)),
 ('nondivisible warp',lambda c:c['warp_shape'].__setitem__(0,7)),
 ('invalid WGMMA N warpgroup',lambda c:(c['block_shape'].__setitem__(1,64),c['warp_shape'].__setitem__(1,32))),
 ('too many threads',lambda c:(c['block_shape'].__setitem__(0,512),c['warp_shape'].__setitem__(0,8))),
 ('insufficient stages',lambda c:c.update(num_stages=2)),
 ('FP16 accum',lambda c:c.update(use_f16_accum=True)),
 ('StreamK',lambda c:c.update(use_stream_k=True)),
 ('nonpositive CTA count',lambda c:c.update(num_ctas_per_sm=0)),
 ('indexed TMA A',lambda c:c.update(use_tma=True,use_tma_a=True)),
 ('layer metadata override',lambda c:c.update(a_dtype='float16')),
 ]:
 x=layer()
 for stage in x.fp32_configs:
  for _,_,c in x.fp32_configs[stage]:c['block_shape']=list(c['block_shape']);c['warp_shape']=list(c['warp_shape']);modify(c)
 check(name+' disables',not prepare(x,torch) and not x._humming_large_prefill_partition_enabled and x._humming_large_prefill_configs==frozenset())
x=layer();props.shared_memory_per_block_optin=1024;props.shared_memory_per_multiprocessor=1024;_get_device_smem_limits.cache_clear();check('device shared memory overflow disables',not prepare(x,torch));props.shared_memory_per_block_optin=227*1024;props.shared_memory_per_multiprocessor=228*1024;_get_device_smem_limits.cache_clear()
x=layer();x.fp32_configs['w2']=[];check('missing eligible second stage disables',not prepare(x,torch))
x=layer();x.w2_post_scale.device=Device('cuda:1');check('mixed devices disable',not prepare(x,torch))
x=layer();x.humming_metas['w13']=N(**{f.name:getattr(x.humming_metas['w13'],f.name) for f in dataclasses.fields(LayerConfig) if f.init});x.humming_metas['w13'].weight_scale_group_size=8;check('unsupported group16 contract disables',not prepare(x,torch))
x=layer();original=sys.modules['humming.utils.device'].fits_device_smem;del sys.modules['humming.utils.device'].fits_device_smem;check('missing installed API disables',not prepare(x,torch));sys.modules['humming.utils.device'].fits_device_smem=original
check('all tests performed no runtime compile load or launch',calls['forbidden']==0)
report={'status':'CPU_PINNED_NATIVE_VALIDATORS_PASS_MOCK_DEVICE','count':len(checks),'checks':checks,'admitted_signatures':len(b._humming_large_prefill_configs),'calls':calls,'fragment_sha256':hashlib.sha256((P/'large_partition_admission.py.txt').read_bytes()).hexdigest(),'source_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [V/'config/config.py',V/'kernel/humming.py',V/'jit/runtime.py',V/'utils/device.py',V/'utils/smem.py',V/'config/mma.py']},'scope':'Real pinned config classes/check_shape/check_dtype/check_scale/check_config/MMA selector/shared-memory estimator. CUDA device, native runtime/compiler dependencies mocked and forbidden; no GPU or compile/numerical qualification.'}
(P/'large_partition_admission_cpu.json').write_text(json.dumps(report,indent=2));print(report['status'],len(checks),'admitted',len(b._humming_large_prefill_configs))
