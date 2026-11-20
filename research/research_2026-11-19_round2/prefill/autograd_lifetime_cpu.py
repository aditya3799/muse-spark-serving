"""CPU proof of frozen row transport's generated-input lifetime."""
import gc,json,weakref
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
cases=[]
for grad_enabled in (True,False):
 for requires_grad in (True,False):
  model=torch.nn.Linear(8,4,bias=False).eval()
  model.weight.requires_grad_(requires_grad)
  with torch.set_grad_enabled(grad_enabled):
   x=torch.randn(16,8)
   input_weak=weakref.ref(x)
   reference=model(x)
   grad_fn=type(reference.grad_fn).__name__ if reference.grad_fn is not None else None
   del x;gc.collect()
   retained=input_weak() is not None
   assert retained==(grad_enabled and requires_grad)
   del reference;gc.collect()
   assert input_weak() is None
  cases.append(dict(grad_enabled=grad_enabled,weight_requires_grad=requires_grad,
      model_eval=not model.training,reference_grad_fn=grad_fn,input_retained_after_del_x=retained))
report=dict(passed=True,scope='Tiny CPU actualTorch lifetime proof, no CUDA allocation. Same ordinary bias-free Linear ownership semantics as row transport qualification.',
  torch_version=str(torch.__version__),cases=cases,
  frozen_risk='Transport reference can retain generated input after del x when startup grad is enabled and FC weight requires_grad. model.eval() does not disable autograd.',
  row16384_extra_gather_while_input_retained=16384*8192*2,
  previously_claimed_bound=1703280640,conservative_corrected_with_retention=1703280640+16384*8192*2)
(HERE/'early_projection_unapplied/autograd_lifetime.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
