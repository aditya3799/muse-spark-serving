"""CPU-only import diagnostic; deliberately never initializes CUDA."""
import faulthandler,time,sys
faulthandler.enable();faulthandler.dump_traceback_later(10,repeat=True)
start=time.monotonic();print('before_torch',sys.version,flush=True)
import torch
print('after_torch',time.monotonic()-start,'cuda_initialized',torch.cuda.is_initialized(),flush=True)
faulthandler.cancel_dump_traceback_later()
