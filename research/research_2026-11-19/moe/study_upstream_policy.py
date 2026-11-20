"""Evaluate pinned upstream SM90 policy for exact NVFP4/BF16 workload, CPU only."""
from pathlib import Path
import sys,json,dataclasses
from types import SimpleNamespace
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here/'upstream_humming'))
import torch
from humming import dtypes
from humming.device import current_device
current_device._value=SimpleNamespace(sm_count=132,is_ppu=False,sm_version=90,name='H200 CPU policy model',sm_major=9,sm_minor=0,max_smem_size=227*1024)
from humming.config import GemmType,LayerConfig
from humming.tune.sm90 import Sm90Heuristics
rows=[]
for n in (4096,8192):
  for m in (8,16,32,64,128,256,512,753,1506,3012,8192,16384,65536,131072):
    for mode in (GemmType.INDEXED,GemmType.GROUPED_CONTIGUOUS):
      layer=LayerConfig(sm_version=90,shape_n=n,shape_k=4096,num_experts=32,a_dtype=dtypes.bfloat16,b_dtype=dtypes.float4e2m1,c_dtype=dtypes.bfloat16,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,input_scale_group_size=0)
      try:
        decision=Sm90Heuristics.get_tuning_decision(layer,m,gemm_type=mode,use_f16_accum=False)
        a=decision.selected_analysis
        rows.append(dict(n=n,k=4096,m=m,gemm_type=mode.value,config=decision.to_config(),reason=decision.reason,analysis=dataclasses.asdict(a),candidate_count=len(decision.considered)))
      except Exception as e:rows.append(dict(n=n,m=m,gemm_type=mode.value,error=repr(e)))
assert not torch.cuda.is_initialized()
(here/'upstream_sm90_policy.json').write_text(json.dumps(dict(commit='0832818a89e2de2b3b672ac1ebd076ba18e0a994',cpu_only=True,rows=rows),indent=2,default=str)+'\n')
print('rows',len(rows),'errors',sum('error' in r for r in rows))
for r in rows:
 if r.get('gemm_type')=='indexed':print(r['n'],r['m'],r.get('config',r.get('error')))
