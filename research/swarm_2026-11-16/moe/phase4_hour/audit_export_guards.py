"""Native CPU audit of exact exported default-off module, no real Torch/CUDA."""
from pathlib import Path
import copy,json,os,sys,types,hashlib
P=Path(__file__).resolve().parent
path=P/'producer_export/candidate_sources_utf8_lf/python/sglang/kernels/ops/moe/muse_spark_humming_producer.py'
ns={};exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),ns)
source=json.loads((P.parent/'sm90_wgmma_metadata_audit.json').read_text(encoding='utf-8'))
class Meta:
 def __init__(self,v):self.value=copy.deepcopy(v)
 def to_str(self):return json.dumps(self.value)
class Layer:
 def __init__(self):
  self.fp32_configs={s:copy.deepcopy(v['intervals']) for s,v in source['stages'].items()}
  self.humming_metas={s:Meta(v['meta']) for s,v in source['stages'].items()}
  self._parameters={'weight':types.SimpleNamespace(device=types.SimpleNamespace(type='cuda'))};self._buffers={}
 def parameters(self):return iter(self._parameters.values())
checks=[];old=os.environ.get('SGLANG_MUSE_HOPPER_PRODUCER');saved={name:sys.modules.get(name) for name in ('torch','humming')}
try:
 for setting in (None,'off','unknown','warp-spec'):
  if setting is None:os.environ.pop('SGLANG_MUSE_HOPPER_PRODUCER',None)
  else:os.environ['SGLANG_MUSE_HOPPER_PRODUCER']=setting
  l=Layer();assert ns['maybe_enable_hopper_producer'](l) is l;checks.append({'name':'default_off_or_unknown','setting':setting,'passed':True})
 calls=[];torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda d:(calls.append(d) or (9,0))))
 humming=types.SimpleNamespace(__version__='0.1.12');sys.modules['torch']=torch;sys.modules['humming']=humming
 for policy in ('b-only','b-scales'):
  os.environ['SGLANG_MUSE_HOPPER_PRODUCER']=policy;l=Layer();original=copy.deepcopy(l.fp32_configs);v=ns['maybe_enable_hopper_producer'](l)
  assert v is not l and v._parameters is l._parameters and v._buffers is l._buffers and l.fp32_configs==original
  assert v._hopper_producer_experiment['enabled'];checks.append({'name':'opt_in_metadata_view','policy':policy,'passed':True})
 for version in ('0.1.11','0.1.13','unknown'):
  humming.__version__=version;l=Layer();assert ns['maybe_enable_hopper_producer'](l) is l;checks.append({'name':'version_fallback','version':version,'passed':True})
 humming.__version__='0.1.12'
 for arch in ((8,6),(10,0),(12,0)):
  torch.cuda.get_device_capability=lambda d,arch=arch:arch;l=Layer();assert ns['maybe_enable_hopper_producer'](l) is l;checks.append({'name':'arch_fallback','arch':arch,'passed':True})
 torch.cuda.get_device_capability=lambda d:(9,0)
 torch.version=types.SimpleNamespace(hip="7.0");l=Layer();assert ns["maybe_enable_hopper_producer"](l) is l;checks.append({"name":"hip_fallback","passed":True});torch.version.hip=None
 for stage,key,value in (('w13','shape_n',4096),('w13','shape_k',8192),('w2','shape_n',8192),('w2','num_experts',64),('w13','pad_shape_n',64),('w2','pad_shape_k',64),('w2','a_dtype','float16')):
  l=Layer();l.humming_metas[stage].value[key]=value;assert ns['maybe_enable_hopper_producer'](l) is l;checks.append({'name':'exact_metadata_fallback','stage':stage,'key':key,'passed':True})
 for bad in (None,'4',True,2):
  l=Layer();l.fp32_configs['w13'][0][2]['num_stages']=bad;assert ns['maybe_enable_hopper_producer'](l) is l;checks.append({'name':'invalid_stage_count_fallback','value':bad,'passed':True})
 l=Layer();l._parameters['weight'].device.type='cpu';assert ns['maybe_enable_hopper_producer'](l) is l;checks.append({'name':'device_fallback','passed':True})
finally:
 if old is None:os.environ.pop('SGLANG_MUSE_HOPPER_PRODUCER',None)
 else:os.environ['SGLANG_MUSE_HOPPER_PRODUCER']=old
 for name,value in saved.items():
  if value is None:sys.modules.pop(name,None)
  else:sys.modules[name]=value
r={'status':'PASS_EXACT_EXPORT_CPU_GUARDS','checks':checks,'helper_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'Exact exported helper under mockedTorch/Humming metadata; noCUDAexecution or performanceproof.'}
(P/'producer_export/cpu_guards.json').write_bytes((json.dumps(r,indent=2)+'\n').encode());print(json.dumps({'status':r['status'],'checks':len(checks)}))
