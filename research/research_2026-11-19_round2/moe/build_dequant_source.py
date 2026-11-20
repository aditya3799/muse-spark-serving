"""Generate a small CABI test from the actual unchanged serving lookup primitive."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent;root=here.parents[1]
old=root/'research_2026-11-18_target_moe/native_package_02/hopper_expert.cu'
s=old.read_text(encoding='utf-8-sig');s=s[:s.index('__device__ __forceinline__ float serving_scaled')]
s+='\n'+(here/'register_dequant.cuh').read_text(encoding='utf-8-sig')
s+=r'''
template<int VARIANT>
__global__ void dequant_probe(const uint4* words, const uint16_t* scales,
                              const uint16_t* table, uint4* out, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  uint4 q = words[index];
  unsigned sf = scales[index];
  unsigned pair = VARIANT ? register_scale_pair(sf) : 0;
  unsigned a[4] = {q.x,q.y,q.z,q.w};
#pragma unroll
  for (int j=0;j<4;++j) {
    uint4 value;
    if constexpr (VARIANT == 0) {
      value = make_uint4(serving_pair<0>(a[j],sf,table),serving_pair<1>(a[j],sf,table),
                         serving_pair<2>(a[j],sf,table),serving_pair<3>(a[j],sf,table));
    } else {
      value = make_uint4(register_pair<0,VARIANT>(a[j],sf,pair,table),register_pair<1,VARIANT>(a[j],sf,pair,table),
                         register_pair<2,VARIANT>(a[j],sf,pair,table),register_pair<3,VARIANT>(a[j],sf,pair,table));
    }
    out[index*4+j] = value;
  }
}
extern "C" int dequant_run(const uint4* words, const uint16_t* scales,
                            const uint16_t* table, uint4* out, int count,
                            int variant, cudaStream_t stream) {
  if (count < 1 || variant < 0 || variant > 2) return int(cudaErrorInvalidValue);
  int grid=(count+255)/256;
  if (variant == 0) dequant_probe<0><<<grid,256,0,stream>>>(words,scales,table,out,count);
  if (variant == 1) dequant_probe<1><<<grid,256,0,stream>>>(words,scales,table,out,count);
  if (variant == 2) dequant_probe<2><<<grid,256,0,stream>>>(words,scales,table,out,count);
  return int(cudaGetLastError());
}
'''
(here/'register_dequant_probe.cu').write_text(s,encoding='utf-8')
(here/'register_dequant_sources.json').write_text(json.dumps(dict(
    incumbent_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256((here/'register_dequant_probe.cu').read_bytes()).hexdigest(),
    ptx_reference='https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html',
    note='No FTZ, no FP8 activation or global-scale change; finite coefficient products exact in CPU proof. Actual CUDA intrinsic and NaN lookup tests passed; see register_dequant_gpu.json.'),indent=2))
print(len(s),hashlib.sha256((here/'register_dequant_probe.cu').read_bytes()).hexdigest())
