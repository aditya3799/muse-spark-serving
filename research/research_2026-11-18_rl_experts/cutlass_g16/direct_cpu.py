"""Compile and execute the exact C++ direct-load fragments on CPU across experts and partial matrix tiles."""
import ctypes as ct
import sys
from direct_vector import VECTOR_BODY
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from direct_loader import W_BODY,S_BODY
from test_exclusive import expected
from test_group_prep import pack,scales

HEADER="""
#include <cstdint>
struct alignas(16) int4 {uint32_t x[4];};
struct alignas(16) uint4 {uint32_t x,y,z,w;};
#define PRAGMA_UNROLL
struct Ctx { static constexpr bool kUseWarpSpec=false,kUseMBarrier=false,kUsePackedKLayout=false; int lane; int load_thread_id() const {return lane;} };
struct ElementA {static constexpr int kBits=16;};
struct ElementB {static constexpr int kBits=4;};
struct ElementBS {static constexpr int kBits=8;};
struct ProblemShape {static constexpr int N=256,K=256;};
template<int BN,int BK> struct Block {static constexpr int N=BN,K=BK;};
template<bool WEIGHT, bool WG, int BN, int BK>
void run(uint8_t const* src, uint8_t* dst,int expert,int nb,int kb) {
  using BlockShape=Block<BN,BK>;
  constexpr int N=ProblemShape::N,K=ProblemShape::K;
  constexpr int kNumLoadThreads=32,kGroupSize=16;
  constexpr bool kUseTma=false,kUseMxScale=false,kIsBlock=false,kIsChannel=false;
  constexpr int kNumInt4s=WEIGHT?BN*BK/32:BN*(BK/16)/16;
  const int4* gmem_ptr_raw=reinterpret_cast<const int4*>(src);
  int4* smem_ptr=reinterpret_cast<int4*>(dst);
  const int4* gmem_ptr=gmem_ptr_raw+(WEIGHT?
      (expert*(K/16)*(N*2)+nb*BN*2+kb*(BK/16)*N*2)/4:
      (expert*(K/16)*N+nb*BN+kb*(BK/16)*N)/16);
  for(int t=0;t<kNumLoadThreads;t++) {
    Ctx ctx{t};
    if constexpr(WEIGHT) {"""
def main():
    result=[]
    vector="--vector" in sys.argv
    weight_body=VECTOR_BODY if vector else W_BODY
    source=HEADER+weight_body.replace("REPLACE_WGMMA","WG")+"} else {"+S_BODY+"}}}\n"
    cases=[(bn,bk,wg,w) for bn,bk in ((64,64),(128,128)) for wg in (0,1) for w in (0,1)]
    for bn,bk,wg,w in cases:
        source+=f'extern "C" void f{bn}_{bk}_{wg}_{w}(uint8_t const*a,uint8_t*b,int e,int n,int k){{run<{w},{wg},{bn},{bk}>(a,b,e,n,k);}}\n'
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);(p/"probe.cpp").write_text(source)
        subprocess.run(["g++","-shared","-fPIC","-std=c++17","-O2",str(p/"probe.cpp"),"-o",str(p/"probe.so")],check=True)
        lib=ct.CDLL(str(p/"probe.so"))
        for seed in (33871,72413):
            rng=np.random.default_rng(seed);e=3;n=k=256;checked=0
            q=rng.integers(16,size=(e,n,k),dtype=np.uint8)
            sf=rng.integers(256,size=(e,n,k//16),dtype=np.uint8)
            qcut,scut=(t.contiguous() for t in expected(q,sf))
            sn=scales(torch.from_numpy(sf)).view(torch.uint8).reshape(e,k//16,n)
            for wg in (0,1):
                wn=pack(q,bool(wg)).view(torch.int32).reshape(e,k//16,2*n)
                for bn,bk,_,weight in (v for v in cases if v[2]==wg):
                    fn=getattr(lib,f"f{bn}_{bk}_{wg}_{weight}")
                    fn.argtypes=(ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int)
                    for expert in (2,0):
                        for nb in range(n//bn):
                            for kb in range(k//bk):
                                if weight:
                                    inp=qcut;gold=wn[expert,kb*(bk//16):(kb+1)*(bk//16),nb*bn*2:(nb+1)*bn*2].contiguous().view(torch.uint8)
                                else:
                                    inp=scut;gold=sn[expert,kb*(bk//16):(kb+1)*(bk//16),nb*bn:(nb+1)*bn].contiguous()
                                out=torch.full_like(gold,0x39)
                                fn(inp.data_ptr(),out.data_ptr(),expert,nb,kb)
                                assert torch.equal(out,gold),(seed,expert,bn,bk,wg,weight,nb,kb)
                                checked+=out.numel()
            result.append(dict(seed=seed,exact_tile_bytes=checked))
    Path(__file__).with_name("direct_vector_cpu.json" if vector else "direct_cpu.json").write_text(json.dumps(result,indent=2))
    print(result)
if __name__=="__main__":main()
