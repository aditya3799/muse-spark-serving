"""Research-only split routing plus fused interleaved W13. Not a serving API."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
OLD=P.parents[1]/'expert_partition'
COORD=P.parents[1]/'coordination'
sys.path.insert(0,str(OLD));sys.path.insert(0,str(COORD))
import scalable_partition
import w13_fused_epilogue
class Composed:
 def __init__(self,layer,routes,global_experts,split=True):
  self.layer=layer
  self.plan=scalable_partition.choose_partition_configs(layer,routes,global_experts) if split else None
  if self.plan is None:
   default={s:next(c for lo,hi,c in intervals if lo<routes<=hi) for s,intervals in layer.fp32_configs.items()}
   self.plan={'single':default}
  self.names=['light','heavy'] if 'light' in self.plan else ['single']
  self.kernels={name:w13_fused_epilogue.prepare(layer,self.plan[name]['w13']) for name in self.names}
 def __call__(self,x,ids,*,defer_downscale=True):
  import torch
  from humming.layer import HummingMethod
  from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
  from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
  layer=self.layer;m,h=x.shape;topk=ids.shape[1];r=ids.numel();e=layer.w13_post_scale.numel()
  if 'light' in self.plan:
   light,heavy,_=scalable_partition.partition_routes_scalable(ids,e,self.plan['cutoff'])
   partitions=[('light',light),('heavy',heavy)]
  else:partitions=[('single',ids)]
  aligned={}
  def align(name,local,bm):
   key=(name,bm)
   if key not in aligned:aligned[key]=moe_align_block_size(local,bm,e,ignore_invalid_expert=True)
   return aligned[key]
  intermediate=layer.humming_metas['w2'].shape_k
  activation=torch.empty((r,intermediate),device=x.device,dtype=x.dtype)
  down=torch.empty((r,h),device=x.device,dtype=x.dtype)
  for name,local in partitions:
   cfg=self.plan[name]['w13']
   self.kernels[name].run(layer,x,activation,local,align(name,local,cfg['block_shape'][0]))
  for name,local in partitions:
   cfg=self.plan[name]['w2'];sorted_ids,expert_ids,padded=align(name,local,cfg['block_shape'][0])
   HummingMethod.forward_layer(layer,inputs=activation,outputs=down,top_k=1,sublayer_name='w2',tuning_config=cfg,sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,valid_shape_m=r,compute_config={'gemm_type':'indexed','use_f16_accum':False})
  if not defer_downscale:scale_active_rows(down,ids,layer.w2_post_scale)
  return down.view(m,topk,h),ids
