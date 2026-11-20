import json,sys
from pathlib import Path
from finite_scale_state import FiniteScaleProof
assert 'torch' not in sys.modules
class Tensor:
 def __init__(self,pointer):self.pointer=pointer;self._version=0;self.dtype='torch.uint8';self.device='cuda:0';self.size=9216
 def data_ptr(self):return self.pointer
 def numel(self):return self.size
class Layer:
 def __init__(self):self.w13=Tensor(1000);self.w2=Tensor(2000)
x=Layer();p=FiniteScaleProof.from_verified_scan(x,True);assert p.matches(x)
checks=[]
for name,modify in [('version',lambda t:setattr(t,'_version',1)),('pointer',lambda t:setattr(t,'pointer',3000)),('size',lambda t:setattr(t,'size',10)),('dtype',lambda t:setattr(t,'dtype','torch.int32')),('device',lambda t:setattr(t,'device','cuda:1'))]:
 x=Layer();p=FiniteScaleProof.from_verified_scan(x,True);modify(x.w2);assert not p.matches(x);checks.append(name)
x=Layer();p=FiniteScaleProof.from_verified_scan(x,True);x.w13=Tensor(1000);assert not p.matches(x);checks.append('replaced_tensor_same_pointer')
assert not p.matches(Layer());checks.append('other_layer')
for finite in (False,None,1,'true'):
 assert not FiniteScaleProof.from_verified_scan(Layer(),finite).verified_finite
 checks.append('unproved_'+repr(finite))
x=Layer();del x.w13._version;assert not FiniteScaleProof.from_verified_scan(x,True).verified_finite;checks.append('missing_version')
class InferenceTensor(Tensor):
 @property
 def _version(self):raise RuntimeError('Inference tensors do not track version counter.')
 @ _version.setter
 def _version(self,value):pass
x=Layer();x.w13=InferenceTensor(1000);assert not FiniteScaleProof.from_verified_scan(x,True).verified_finite;checks.append('inference_version_unavailable')
report=dict(complete=True,cpu_import_without_torch=True,negative_guards=checks,static_graph_weights_required=True,not_runtime_enabled=True)
Path(__file__).with_name('finite_scale_state_cpu.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
