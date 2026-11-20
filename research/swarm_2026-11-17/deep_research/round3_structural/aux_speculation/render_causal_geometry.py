"""CPU-only matched causal geometry/source contract; no CUDA context."""
from pathlib import Path
import sys,json,inspect,textwrap,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parents[3];sys.path.insert(0,str(ROOT/'swarm_2026-11-16/moe/humming_vendor'))
import torch
from humming import dtypes
from humming.config import LayerConfig,ComputeConfig,TuningConfig
from humming.kernel.humming import HummingKernel
class Kernel(HummingKernel):
 def __new__(cls,*a,**kw):return object.__new__(cls)
 def __post_init__(self):
  LayerConfig.__post_init__(self);ComputeConfig.__post_init__(self);TuningConfig.__post_init__(self);self.sm_version=86;self.sm_version_str='86';self.init_kernel()
 def prepare(self):assert not torch.cuda.is_initialized()
 def load_cubin(self):raise AssertionError('No load')
 @staticmethod
 def current_context():raise AssertionError('No context')
 @staticmethod
 def _ensure_cuda_context():raise AssertionError('No context')
code=textwrap.dedent(inspect.getsource(HummingKernel.check_dtype));cap='assert self.b_dtype.num_bits <= 8';assert code.count(cap)==1;code=code.replace(cap,'assert self.b_dtype.num_bits <= 8 or (self.a_dtype==self.b_dtype==self.c_dtype==self.bs_dtype==dtypes.bfloat16 and self.is_channel_weight_scale and not self.has_zero_point and not self.has_bias and not self.use_f16_accum)');ns={'dtypes':dtypes};exec(code,ns);Kernel.check_dtype=ns['check_dtype']
old=json.loads((P/'packed_native_02/results.json').read_text());tablepath=ROOT/'swarm_2026-11-17/deep_research/expert_partition/k_preserve_m1024_01/owned_configs.json';table=json.loads(tablepath.read_text())['indexed'];cases=[]
for stage in ('w13','w2'):
 meta=next(v['original_config'] for v in old['kernels'] if v['stage']==stage);c=next(v for lo,hi,v in table[stage] if lo<4096<=hi);meta.update(c);meta.pop('num_sms',None)
 matched=json.loads(json.dumps(meta));matched['block_shape']=[16,128,64];matched['warp_shape']=[16,64,64]
 b16=dict(matched,shape_n=128,b_dtype='bfloat16',bs_dtype='bfloat16',weight_scale_type='channel',weight_scale_group_size=0,num_stages=2)
 for name,cfg in (('production_W4',meta),('matched_W4',matched),('tiled_B16',b16)):
  k=Kernel(**cfg);mma=k.mma_op_class.to_cpp_str();cases.append({'stage':stage,'variant':name,'config':cfg,'mma_sha256':hashlib.sha256(mma.encode()).hexdigest(),'estimated_shared_bytes':k.estimated_smem_size,'threads':k.num_threads,'logical_M_tiles':(512+k.block_shape[0]-1)//k.block_shape[0],'logical_N_tiles':k.shape_n//k.block_shape[1],'physical_grid':'Runtime scheduler not inspected here; logicaltiles are not residentCTAcount.'})
 assert len({v['mma_sha256'] for v in cases if v['stage']==stage})==1
assert not torch.cuda.is_initialized()
r={'status':'CPU_CAUSAL_GEOMETRY_RENDER_PASS','cases':cases,'scope':'ChangedM/BN but identicalbaseWMMAprimitive andKtopology; numerical/runtimeproof pending. No compiler/CUDAinitialization.','source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),P/'probe_output_n_causal.py',P/'native_materialize_tiled.py',P/'native_materialize_tiled.cuh',P/'output_n_band_ops.py',tablepath)}}
(P/'OUTPUT_N_CAUSAL_CPU_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'cases':[{k:c[k] for k in ('stage','variant','estimated_shared_bytes','threads','logical_M_tiles','logical_N_tiles')} for c in cases]}))
