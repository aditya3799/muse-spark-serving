from pathlib import Path
import sys,json
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p/'humming_vendor'))
import torch
from humming import dtypes
from humming.config import WeightScale2Type,WeightScaleType,GemmType
from humming.layer import HummingMethod
from humming.schema import HummingWeightSchema,HummingInputSchema
from humming.tune.sm90 import Sm90Heuristics
layer=torch.nn.Module();report={}
for stage,n in [('w13',8192),('w2',4096)]:
 schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE)
 meta=HummingMethod.prepare_layer_meta(layer,shape_n=n,shape_k=4096,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=32,torch_dtype=torch.bfloat16,sublayer_name=stage)
 cfg=Sm90Heuristics.get_configs(meta,use_f16_accum=False,gemm_type=GemmType.INDEXED)
 for lo,hi,c in cfg:c['use_stream_k']=False
 report[stage]={'meta':meta.to_str(),'intervals':cfg}
(p/'sm90_tuning_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({s:len(r['intervals']) for s,r in report.items()}))
for m in [1,8,16,32,33,128,256,257,512,1024,4096]:
 row={'tokens':m}
 for stage in report:
  c=next(c for lo,hi,c in report[stage]['intervals'] if lo<m*8<=hi)
  row[stage]=c
 print(json.dumps(row))
