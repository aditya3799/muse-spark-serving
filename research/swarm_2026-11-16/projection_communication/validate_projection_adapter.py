from pathlib import Path
import hashlib,importlib.util,json,sys
import torch
P=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/projection_communication')
R=Path('/mnt/d/OpenKernelChallenge/muse_bf16_proj')
sys.path.insert(0,str(R))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ref=load('proj_reference',R/'reference.py')
impl=load('projection_h200_adapter',P/'projection_h200_adapter.py')
cases=[(1,'qkv_gate','normal'),(7,'pooler','normal'),(33,'o','normal'),(130,'unpooler','normal'),(5,'unpooler','normal'),(8,'router','normal'),(9,'router','outliers'),(512,'qkv_gate','outliers'),(1000,'unpooler','outliers')]
rows=[]
with torch.inference_mode():
 weights=ref.generate_weights(6382)
 state=impl.prepare(weights)
 for i,(m,layer,distribution) in enumerate(cases):
  data=ref.generate_input(m,layer,seed=731+i,dist=distribution)
  out=impl.kernel_fn(data,state)
  ok,message=ref.check_implementation(data,out,weights)
  rows.append(dict(m=m,layer=layer,distribution=distribution,passed=ok,message=message))
  print(json.dumps(rows[-1]),flush=True)
  assert ok,message
 # Nearby geometry/data refresh: changed seeds and both BF16 reduction flags.
 for reduction in (False,True):
  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduction
  for m in (8,16,32,128,256):
   for layer in ('pooler','unpooler'):
    data=ref.generate_input(m,layer,seed=929+m,dist='outliers')
    out=impl.kernel_fn(data,state)
    ok,message=ref.check_implementation(data,out,weights)
    rows.append(dict(m=m,layer=layer,distribution='outliers',reduced_precision_flag=reduction,passed=ok,message=message))
    assert ok,message
report=dict(passed=all(r['passed'] for r in rows),official_cases=9,additional_cases=20,rows=rows,sha256=hashlib.sha256((P/'projection_h200_adapter.py').read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),scope='Full-output adapter checked against public FP64 reference and probe-row checks locally; remote H200 hidden data, network, serving behavior unvalidated.')
(P/'projection_adapter_local_validation.json').write_text(json.dumps(report,indent=2));print('All29 adapter cases passed',flush=True)
