"""Exact cancellation fixture distinguishing WK32 from accidentally retained WK64."""
def set_engineered(layer,canonical,x,n,k):
 import torch
 from humming import dtypes
 from humming.config import WeightScaleType,WeightScale2Type
 from humming.layer import HummingMethod
 from humming.schema import HummingWeightSchema,HummingInputSchema
 assert layer.weight.shape[0]==1 and k==4096
 raw=torch.zeros((1,n,k//2),device=x.device,dtype=torch.uint8)
 raw[:,:,0]=6;raw[:,:,16]=6;raw[:,:,32]=6;raw[:,:,48]=6
 sf=torch.ones((1,n,k//16),device=x.device,dtype=torch.float32).to(torch.float8_e4m3fn)
 new=torch.nn.Module();new.register_parameter('weight',torch.nn.Parameter(raw.view(torch.int32),requires_grad=False));new.register_parameter('weight_scale',torch.nn.Parameter(sf,requires_grad=False))
 schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE)
 HummingMethod.prepare_layer_meta(new,shape_n=n,shape_k=k,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=1,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(new,already_padded=True)
 word=new.weight.to(torch.int64)&0xffffffff;converted=((word&0xf00ff00f)|((word&0x00f000f0)<<4)|((word&0x0f000f00)>>4)).to(torch.int32)
 layer.weight.copy_(new.weight);layer.weight_scale.copy_(new.weight_scale);canonical.copy_(converted)
 x.zero_();x[:,0]=2.**24;x[:,32]=1.;x[:,64]=-2.**24;x[:,96]=1.
 return dict(expected_fp32=2.,expected_bf16_x4=8.,wrong_old_wk64_fp32=1.,positions=[0,32,64,96],full_dot_values=[2**24,1,-2**24,1])
