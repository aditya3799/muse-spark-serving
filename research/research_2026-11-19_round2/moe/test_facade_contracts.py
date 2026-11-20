"""Cheap import/metadata/lifecycle negative gates, without Torch or CUDA."""
import ast
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from hybrid_facade import SharedWeightWorkspace,validate_resident

here=Path(__file__).resolve().parent
assert 'torch' not in sys.modules
T=NS(uint8='u8',bfloat16='bf16')
device=NS(type='cuda')
class Tensor:
    def __init__(self,size,dtype,shape,dev=device,contiguous=True):
        self.size=size;self.dtype=dtype;self.shape=shape;self.device=dev;self.contiguous=contiguous
    def numel(self):return self.size
    def is_contiguous(self):return self.contiguous

def resident(e=32):
    return NS(experts=e,w13=Tensor(e*8192*2304,'u8',(1,)),w2=Tensor(e*4096*2304,'u8',(1,)),
              w13_post_scale=Tensor(e,'bf16',(e,)),w2_post_scale=Tensor(e,'bf16',(e,)))
passed=[]
for e in (1,2,32):assert validate_resident(resident(e),T)==(e,device)
for attr,value in [('experts',0),('experts',33),('experts',True),
                  ('w13',Tensor(1,'u8',(1,))),('w2',Tensor(32*4096*2304,'bf16',(1,))),
                  ('w13_post_scale',Tensor(32,'u8',(32,))),
                  ('w2_post_scale',Tensor(32,'bf16',(1,32))),
                  ('w2',Tensor(32*4096*2304,'u8',(1,),contiguous=False))]:
    r=resident();setattr(r,attr,value)
    try:validate_resident(r,T)
    except ValueError:passed.append(attr)
    else:raise AssertionError(attr)
state=NS(stream=7,capturing=False)
w=SharedWeightWorkspace.__new__(SharedWeightWorkspace)
w.owner_thread=threading.get_ident();w.closed=False;w.device=device;w.stream=7
w.torch=NS(cuda=NS(current_stream=lambda dev:state.stream,is_current_stream_capturing=lambda:state.capturing))
w.facades={};w.sealed=False
w._check_context()
for kind in ('thread','closed','stream'):
    old=(w.owner_thread,w.closed,state.stream)
    if kind=='thread':w.owner_thread=-1
    if kind=='closed':w.closed=True
    if kind=='stream':state.stream=8
    try:w._check_context()
    except RuntimeError:passed.append(kind)
    else:raise AssertionError(kind)
    w.owner_thread,w.closed,state.stream=old
w.seal_for_capture()
try:w.close()
except RuntimeError:passed.append('graph_owned_close')
else:raise AssertionError('closed graph-owned arena')
class Unknown:pass
try:w.prepare(Unknown())
except RuntimeError:passed.append('new_metadata_after_seal')
else:raise AssertionError('prepared metadata after seal')
try:w.convert(Unknown())
except RuntimeError:passed.append('unprepared_resident')
else:raise AssertionError('unprepared conversion')
assert 'torch' not in sys.modules and 'humming' not in sys.modules
source=ast.parse((here/'hybrid_facade.py').read_text(encoding='utf-8-sig'))
for node in ast.walk(source):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
        assert node.func.attr not in ('transform_humming_layer','transform_humming_tensors')
report=dict(complete=True,torch_free_import=True,no_runtime_weight_transform=True,
            negatives=passed,positive_experts=[1,2,32],shared_packed_bytes_32=32*12288*2304,
            quantized_layer_traffic={'per_layer_repack_bytes':2*32*12288*2304,
              'sixty_quantized_layers_if_verified':60*2*32*12288*2304,
              'sixty_two_MoE_layers_if_all_quantized':62*2*32*12288*2304,
              'endpoint_dtype_not_inferred':True})
(here/'facade_contracts.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
