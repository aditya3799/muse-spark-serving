import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
from producer_layer_view import producer_layer_view
P=Path(__file__).resolve().parent
source=json.loads((P.parent/'sm90_wgmma_metadata_audit.json').read_text(encoding='utf-8'))
class Meta:
 def __init__(self,value):self.value=value
 def to_str(self):return json.dumps(self.value)
def layer():
 return NS(fp32_configs={s:copy.deepcopy(v['intervals']) for s,v in source['stages'].items()},humming_metas={s:Meta(copy.deepcopy(v['meta'])) for s,v in source['stages'].items()},_parameters={'weight':object()},_buffers={'scale':object()})
checks=[]
for policy in ('tma_b_only','tma_b'):
 l=layer();original=copy.deepcopy(l.fp32_configs);view,report=producer_layer_view(l,policy,90)
 assert report['enabled'] and view is not l
 assert view._parameters is l._parameters and view._buffers is l._buffers
 assert l.fp32_configs==original
 for stage in original:
  for (_,_,c),(_,_,o) in zip(view.fp32_configs[stage],original[stage]):
   assert c['block_shape']==o['block_shape'] and c['warp_shape']==o['warp_shape'] and c['num_stages']==o['num_stages']
   assert not c['use_stream_k'] and not c['use_f16_accum'] and not c['use_warp_spec']
   assert c['use_tma_b'] and c['use_tma_bs']==(policy=='tma_b')
 view.fp32_configs['w13'][0][2]['block_shape'][0]+=8;assert l.fp32_configs==original
 checks.append({'name':'shared_tensors_owned_configs','policy':policy,'passed':True})
for arch in (75,80,86,89,100,120):
 l=layer();v,r=producer_layer_view(l,'tma_b_only',arch);assert v is l and not r['enabled'];checks.append({'name':'architecture_fallback','arch':arch,'passed':True})
for policy in ('warp_spec','both','warp_tma_b_only','unknown'):
 l=layer();v,r=producer_layer_view(l,policy,90);assert v is l and not r['enabled'];checks.append({'name':'policy_fallback','policy':policy,'passed':True})
for field,value in (('a_dtype','float16'),('b_dtype','int4'),('weight_scale_group_size',32),('use_int_weight_scale',True),('has_zero_point',True),('has_bias',True),('use_packed_k_layout',True),('has_input_scale',True)):
 l=layer();l.humming_metas['w2'].value[field]=value;v,r=producer_layer_view(l,'tma_b_only',90);assert v is l and not r['enabled'];checks.append({'name':'metadata_fallback','field':field,'passed':True})
for field in ('use_stream_k','use_f16_accum'):
 l=layer();l.fp32_configs['w13'][0][2][field]=True;v,r=producer_layer_view(l,'tma_b_only',90);assert v is l and not r['enabled'];checks.append({'name':'arithmetic_config_fallback','field':field,'passed':True})
r={'status':'PASS_CPU_MOCK_METADATA_GUARDS','checks':checks,'scope':'NoTorch/GPU. Verifiesmetadatafallbackandcopyownership; actualCUDAparameterpointerscheckedinpreparedHopperharness.','source_hashes':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ('producer_tuning.py','producer_layer_view.py','producer_fullchain_probe.py','audit_producer_view.py')}}
(P/'producer_view_cpu_result.json').write_bytes((json.dumps(r,indent=2)+'\n').encode());print(json.dumps({'status':r['status'],'checks':len(checks)}))
