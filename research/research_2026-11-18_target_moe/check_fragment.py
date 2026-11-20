"""Run the native decoder and activation on the available GPU against independent references."""
import argparse
import ast
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys
import torch
from qualify_packing import DONOR, ROOT

PROBE = r"""
__global__ void check(const unsigned* q, const unsigned* s, const uint16_t* lut,
                      unsigned* y, const float* accum, const float* scales,
                      __nv_bfloat16* activation, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  y[4*i] = serving_pair<0>(q[i], s[i], lut);
  y[4*i+1] = serving_pair<1>(q[i], s[i], lut);
  y[4*i+2] = serving_pair<2>(q[i], s[i], lut);
  y[4*i+3] = serving_pair<3>(q[i], s[i], lut);
  const int row = i/4096, col = i%4096;
  activation[i] = serving_activation(accum[row*8192+col], accum[row*8192+4096+col],scales[row]);
}
extern "C" int run(const unsigned* q, const unsigned* s, const uint16_t* lut, unsigned* y,
                   const float* a, const float* scales, __nv_bfloat16* act, int n, cudaStream_t stream) {
  check<<<(n+255)/256,256,0,stream>>>(q,s,lut,y,a,scales,act,n);
  return cudaGetLastError();
}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    sys.path.insert(0, str(ROOT/"optimization_marlin_tiles_2026-11-15"))
    from kernel_runtime import bootstrap
    bootstrap(ROOT/"sglang")
    from sglang.kernels.ops.activation.activation import silu_and_mul
    arch = "".join(map(str,torch.cuda.get_device_capability()))
    source = out/"fragment.cu"
    source.write_text('#include <cuda_runtime.h>\n' + Path(__file__).with_name("serving_fragment.cuh").read_text() + PROBE)
    lib = out/"libfragment.so"
    subprocess.run(["nvcc","-O3","-std=c++17",f"-arch=sm_{arch}","-shared","-Xcompiler=-fPIC","--cudart=shared",str(source),"-o",str(lib)],check=True)
    run = ct.CDLL(str(lib)).run
    run.argtypes = [ct.c_void_p]*7 + [ct.c_int,ct.c_void_p]
    torch.manual_seed(1311)
    m, k = 64,4096
    idx = torch.arange(m*k,device="cuda")
    # Cover the full code x scale table independently in each register half.
    code = torch.stack([((idx//256)*(2*j+1)+j+idx//4096)%16 for j in range(8)],dim=1)
    lo,hi = idx%256,(idx*19+17)%256
    scales = (lo | hi<<8).int()
    node = next(n for n in ast.parse(DONOR.read_text()).body if getattr(n,"name",None)=="_stored_pos_mul")
    ns = {}
    exec(compile(ast.Module([node],[]),str(DONOR),"exec"),ns)
    positions = ns["_stored_pos_mul"]()
    words = torch.zeros_like(idx)
    for (r,h,b),pos in positions.items():
        words |= (code[:,2*r+h]>>b & 1)<<pos
    words = words.int()
    fp4 = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device="cuda")
    fp8 = torch.arange(256,device="cuda").byte().view(torch.float8_e4m3fn).float()
    lut = (fp8[:,None]*fp4).bfloat16().contiguous()
    actual = torch.empty((m*k,8),device="cuda",dtype=torch.int16)
    expected = torch.stack([(fp8[hi if j//2%2 else lo]*fp4[code[:,j]]).bfloat16().view(torch.int16) for j in range(8)],dim=1)
    a = torch.randn((m,2*k),device="cuda")*3
    global_scales = torch.linspace(.125,1.5,m,device="cuda").bfloat16().float()
    act = torch.empty((m,k),device="cuda",dtype=torch.bfloat16)
    status = run(*[t.data_ptr() for t in (words,scales,lut,actual,a,global_scales,act)],m*k,torch.cuda.current_stream().cuda_stream)
    if status: raise RuntimeError(status)
    scaled = (a.bfloat16().float()*global_scales[:,None]).bfloat16()
    target = torch.empty_like(act)
    silu_and_mul(scaled,target,expert_ids=torch.arange(m,device="cuda",dtype=torch.int32),expert_step=1)
    data = dict(device=torch.cuda.get_device_name(),decode_values=expected.numel(),
                decode_mismatches=int((expected!=actual).sum()),activation_values=act.numel(),
                activation_mismatches=int((act!=target).sum()),activation_max_abs=float((act.float()-target.float()).abs().max()))
    (out/"results.json").write_text(json.dumps(data,indent=2))
    print(data)
    assert data["decode_mismatches"] == data["activation_mismatches"] == 0

if __name__ == "__main__":
    main()
