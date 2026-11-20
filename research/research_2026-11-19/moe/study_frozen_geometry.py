from pathlib import Path
import sys,json
here=Path(__file__).resolve().parent;root=here.parents[1]
sys.path.insert(0,str(root/'swarm_2026-11-16/moe/humming_vendor'))
import torch
from humming import dtypes
from humming.config import LayerConfig,GemmType,MmaType
from humming.tune.sm90 import Sm90Heuristics
rows=[]
for n in (8192,4096):
 layer=LayerConfig(shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,c_dtype=dtypes.bfloat16,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,mma_type=MmaType.WGMMA)
 table=Sm90Heuristics.get_configs(layer,use_f16_accum=False,gemm_type=GemmType.INDEXED)
 for m in (8,32,128,256,753,3012,8192,16384):
  configs={}
  for tag,q in [('light',m),('heavy',8*m)]:
   c=next(c for lo,hi,c in table if lo<q<=hi).copy();c['use_stream_k']=False
   configs[tag]=c
  # Frozen large-prefill selector forces light K geometry to match heavy.
  ratio=configs['heavy']['block_shape'][2]//configs['heavy']['warp_shape'][2]
  rows.append(dict(n=n,m=m,configs=configs,frozen_K_accumulator_chains=ratio,single_chain_grouped_math_possible=ratio==1))
assert not torch.cuda.is_initialized()
(here/'frozen_sm90_k_geometry.json').write_text(json.dumps(dict(cpu_only=True,rows=rows),indent=2))
for r in rows:print(r['n'],r['m'],r['frozen_K_accumulator_chains'],r['configs']['heavy']['block_shape'],r['configs']['heavy']['warp_shape'])
