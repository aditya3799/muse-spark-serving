"""Single-layout Hopper routed expert projections for the Muse NVFP4 backend."""
import ctypes as ct
import logging
from functools import lru_cache
from pathlib import Path

import torch

from .hopper_pack import pack_experts


@lru_cache(maxsize=1)
def _library():
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("hopper_expert.cu")
    path = load(
        name="sglang_muse_hopper_expert",
        sources=[str(source)],
        extra_cuda_cflags=[
            "-O3", "-gencode=arch=compute_90a,code=sm_90a",
            "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
            "-DOKC_KPERM3=1", "-DOKC_PRMTSC=1", "-DOKC_NS=3", "-DOKC_NSF=3", "-DOKC_FNST=4",
        ],
        is_python_module=False,
    )
    lib = ct.CDLL(path)
    lib.expert_config.argtypes = [ct.POINTER(ct.c_int)]
    lib.expert_sizes.argtypes = [ct.POINTER(ct.c_size_t)]
    lib.expert_sizes.restype = None
    lib.expert_run.argtypes = [ct.c_void_p]*10 + [ct.c_int]*3 + [ct.c_void_p]
    return lib


@lru_cache(maxsize=None)
def _log_selection(device, grid, status):
    logging.getLogger(__name__).info("Muse Hopper routed experts on %s, grid %d, CUDA configuration status %d", device, grid, status)


class HopperExperts(torch.nn.Module):
    @classmethod
    def prepare(cls, layer, raw):
        device = raw.w13_weight.device
        if (device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0)
                or layer.top_k != 8 or layer.params_dtype != torch.bfloat16
                or layer.hidden_size != 4096 or layer.intermediate_size_per_partition != 4096
                or not 1 <= layer.num_local_experts <= 32):
            return None
        with torch.cuda.device(device):
            lib = _library()
            grid = ct.c_int()
            status = lib.expert_config(ct.byref(grid))
            _log_selection(device, grid.value, status)
            if status:
                return None
            sizes = (ct.c_size_t * 3)()
            lib.expert_sizes(sizes)
            result = cls()
            result.lib = lib
            result.grid = grid.value
            result.sizes = tuple(sizes)
            result.experts = layer.num_local_experts
            for stage in ("w13","w2"):
                scale = getattr(raw,stage+"_weight_scale_2")
                if scale.shape != (result.experts,):
                    raise ValueError("expected one global scale per expert")
                scale = scale.to(torch.bfloat16).contiguous()
                result.register_buffer(stage+"_post_scale",scale)
                result.register_buffer(stage,pack_experts(getattr(raw,stage+"_weight"),
                    getattr(raw,stage+"_weight_scale"),stage=="w13"))
            result.register_buffer("gate_scale",result.w13_post_scale.float())
            fp4 = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=device)
            fp8 = torch.arange(256,device=device).byte().view(torch.float8_e4m3fn).float()
            result.register_buffer("table",(fp8[:,None]*fp4).bfloat16())
            return result

    def project(self, x, ids, defer_downscale):
        from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows

        rows = x.size(0)
        if (x.shape != (rows,4096) or ids.shape != (rows,8) or x.dtype != torch.bfloat16
                or ids.dtype != torch.int32 or x.device != self.w13.device or x.device != ids.device):
            raise ValueError("invalid input to Hopper routed experts")
        x = x.contiguous()
        ids = ids.contiguous()
        out = x.new_empty((rows,8,4096))
        if not rows:
            return out,ids
        with torch.cuda.device(x.device):
            counter = torch.zeros(self.sizes[0],device=x.device,dtype=torch.int32)
            ready = torch.zeros(self.sizes[1],device=x.device,dtype=torch.int32)
            tiles = x.new_empty(self.sizes[2])
            stream = torch.cuda.current_stream(x.device).cuda_stream
            for first in range(0,rows,384):
                size = min(rows-first,384)
                args = (self.w13,self.w2,self.gate_scale,self.table,x[first],ids[first],out[first],counter,ready,tiles)
                status = self.lib.expert_run(*(a.data_ptr() for a in args),size,self.experts,self.grid,stream)
                if status:
                    raise RuntimeError(f"Hopper expert launch failed with CUDA error {status}")
            if not defer_downscale:
                scale_active_rows(out.view(-1,4096),ids,self.w2_post_scale)
        return out,ids
