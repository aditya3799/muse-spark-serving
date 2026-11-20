"""Generate one-GPU virtual-rank qualification wrappers; no compilation here."""
import hashlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent
ROOT=next(p for p in P.parents if (p/'RULES.md').exists())
def body(src,name):
 i=src.index(name);a=src.index('{',i);level=1;j=a+1
 while level:
  level+=(src[j]=='{')-(src[j]=='}');j+=1
 return src[a+1:j-1]
def main():
 source=(P/'attention_ar_norm.cuh').read_text(encoding='utf-8-sig')
 norm=(ROOT/'sglang/python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh').read_text()
 constants='''
  constexpr uint32_t kWorld=8;
  constexpr int64_t kHidden=8192;
  constexpr int kNumThreads=256;
  constexpr bool kUsePDL=false, kPrefetchWeights=true, kPostNormIsInputScale=false;
  constexpr bool kHasRouter=true, kPostWeight=true, kGate=true;
'''
 fused=body(source,'research_attention_ar_norm_kernel(')
 # No changing thread/block indices or expression ordering in the original math body.
 fused=constants+'  const auto& collective = all_params[blockIdx.y];\n'+fused
 fused=fused.replace('    bool pending;', '    bool pending;\n    const unsigned long long debug_wait_start=clock64();').replace('    } while (pending);', '      if (pending && clock64()-debug_wait_start > 4000000000ull) asm volatile('+chr(34)+'trap;'+chr(34)+');\n    } while (pending);')
 ref=constants+'  const auto& params = all_params[blockIdx.y];\n'+body(norm,'muse_spark_norm_residual_kernel(')
 prefix='''#include "attention_ar_norm.cuh"
#include <cstddef>
using namespace sglang;
extern "C" __global__ void abi_sizes(unsigned long long* out) {
 if(threadIdx.x==0){out[0]=sizeof(NormResidualParams);out[1]=sizeof(device::distributed::PushWorkSpace<8>);out[2]=sizeof(ResearchAttentionARParams<8>);out[3]=offsetof(ResearchAttentionARParams<8>,rank);out[4]=offsetof(NormResidualParams,pre_norm_bf16);}
}
extern "C" __global__ void rank_order_reference(const bf16_t* partials,bf16_t* out,int m) {
 using namespace device;
 using V=AlignedVector<bf16x2_t,4>;
 using L=distributed::LamportTrait<bf16_t,8,4>;
 int v=blockIdx.x*blockDim.x+threadIdx.x;
 int count=m*8192/8;
 if(v>=count)return;
 V peers[8];
 for(int r=0;r<8;r++){peers[r].load(partials+r*m*8192,v);L::clear_pos_zero(peers[r].data());}
 V reduced=reduce_vec(peers);
 for(int r=0;r<8;r++)reduced.store(out+r*m*8192,v);
}
'''
 code=prefix+'extern "C" __global__ __launch_bounds__(256) void virtual_fused(const ResearchAttentionARParams<8>* all_params) {'+fused+'}\n'
 code+='extern "C" __global__ __launch_bounds__(256) void virtual_reference_norm(const NormResidualParams* all_params) {'+ref+'}\n'
 # Prefilled mode removes ONLY own producer loop, allowing M8 math test without scheduling a waiting grid.
 start=fused.index('  // Every row\'s producer')
 end=fused.index('  BF16Vec zero;',start)
 prefilled=fused[:start]+fused[end:]
 code+='extern "C" __global__ __launch_bounds__(256) void virtual_prefilled_consumer(const ResearchAttentionARParams<8>* all_params) {'+prefilled+'}\n'
 code+='''extern "C" __global__ void prefill_planes(const bf16_t* partials,unsigned char* plane,int m,int slot_bytes) {
 using namespace device;using V=AlignedVector<bf16x2_t,4>;using L=distributed::LamportTrait<bf16_t,8,4>;
 int vid=blockIdx.x*blockDim.x+threadIdx.x;int count=m*8192/8;
 if(vid>=8*count)return;int src=vid/count;int v=vid%count;V value;value.load(partials+src*m*8192,v);L::clear_pos_zero(value.data());
 for(int dst=0;dst<8;dst++)value.store(reinterpret_cast<bf16_t*>(plane+dst*16*slot_bytes+src*slot_bytes),v);
}
'''
 (P/'virtual_attention_ar.cu').write_bytes(code.encode())
 meta=dict(status='GENERATED_ONLY',source_sha256=hashlib.sha256(code.encode()).hexdigest(),modes=dict(virtual_fused='M1 only,8CTAs cooperative launch, full producer/poll/clear/flip; singleGPU memory only',virtual_prefilled_consumer='M1..8, planesfilled in completedprior kernel, no concurrentproducers/protocol timing claim',reference='sameLamport sanitize+reduce_vec then exactsource singleCTA norm'),constants=dict(W=8,H=8192,threads=256,post_weight=True,gate=True,router=True,PDL=False),memory_estimate_M8='less than64MiB tensors',limitations=['Not actual multiGPU addresses, caches, NVLink or H200','No timing promotion','Source does not install a live serving hook'])
 (P/'VIRTUAL_FIXTURE_CONTRACT.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()

