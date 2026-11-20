"""Pinned-source SM90 metadata audit; architecture interfaces mocked, no kernels."""
from pathlib import Path
import sys,types,importlib.util,json,hashlib
p=Path(__file__).resolve().parent;vendor=p/'humming_vendor/humming'
for name in ['humming','humming.config','humming.tune','humming.utils']:
 mod=types.ModuleType(name);mod.__path__=[];sys.modules[name]=mod
fake=types.ModuleType('torch');fake.cuda=types.SimpleNamespace(get_device_capability=lambda:(9,0))
for name in ['float8_e8m0fnu','float8_e4m3fn','float8_e5m2','float16','bfloat16','float32']:
 setattr(fake,name,'torch.'+name)
sys.modules['torch']=fake
hashes={}
def load(name,rel):
 path=vendor/rel;hashes[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
 sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
d=load('humming.dtypes','dtypes.py');sys.modules['humming'].dtypes=d
enums=load('humming.config.enum','config/enum.py')
load('humming.config.base','config/base.py')
config=load('humming.config.config','config/config.py')
for name in ['LayerConfig','ComputeConfig','TuningConfig']:
 setattr(sys.modules['humming.config'],name,getattr(config,name))
for name in ['GemmType','MmaType','WeightScaleType','WeightScale2Type']:
 setattr(sys.modules['humming.config'],name,getattr(enums,name))
def unused(*args,**kwargs):raise AssertionError('Unexpected hardware-dependent helper in SM90 heuristic audit')
for name,attr in [('humming.utils.device','get_device_num_sms'),('humming.utils.smem','estimate_smem_size_layer')]:
 mod=types.ModuleType(name);setattr(mod,attr,unused);sys.modules[name]=mod
load('humming.tune.base','tune/base.py');sm90=load('humming.tune.sm90','tune/sm90.py')
report={'scope':'Architecture-mocked pinned-source metadata/heuristics only. No WGMMA runtime compilation or hardware proof.','source_hashes':hashes,'stages':{}}
for stage,n in [('w13',8192),('w2',4096)]:
 meta=config.LayerConfig(shape_n=n,shape_k=4096,num_experts=32,a_dtype=d.bfloat16,b_dtype=d.float4e2m1,c_dtype=d.bfloat16,bs_dtype=d.float8e4m3,weight_scale_group_size=16,weight_scale_type=enums.WeightScaleType.GROUP,weight_scale_2_type=enums.WeightScale2Type.NONE)
 assert meta.mma_type==enums.MmaType.WGMMA and not meta.has_input_scale and not meta.use_int_weight_scale
 intervals=sm90.Sm90Heuristics.get_configs(meta,use_f16_accum=False,gemm_type=enums.GemmType.INDEXED)
 for lo,hi,c in intervals:c['use_stream_k']=False
 report['stages'][stage]={'meta':json.loads(meta.to_str()),'intervals':intervals}
previous=json.loads((p/'sm90_tuning_audit.json').read_text())
report['same_intervals_as_earlier_shape_audit']=all(json.loads(json.dumps(report['stages'][s]['intervals']))==previous[s]['intervals'] for s in previous)
(p/'sm90_wgmma_metadata_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'passed':True,'stages':{s:{'mma_type':r['meta']['mma_type'],'interval_count':len(r['intervals'])} for s,r in report['stages'].items()},'same_intervals':report['same_intervals_as_earlier_shape_audit']}))

