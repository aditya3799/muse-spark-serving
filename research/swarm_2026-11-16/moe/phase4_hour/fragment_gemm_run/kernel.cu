#include "/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/moe/phase4_hour/fragment_gemm.cuh"
extern "C" __global__ void gemm0(const uint32_t*c,const uint8_t*s,const nv_bfloat16*x,float*a,nv_bfloat16*r,nv_bfloat16*f,uint32_t M,uint32_t N,uint32_t K,uint16_t gs){fragment_gemm::run<false>(c,s,x,a,r,f,M,N,K,*reinterpret_cast<nv_bfloat16*>(&gs));}
extern "C" __global__ void gemm1(const uint32_t*c,const uint8_t*s,const nv_bfloat16*x,float*a,nv_bfloat16*r,nv_bfloat16*f,uint32_t M,uint32_t N,uint32_t K,uint16_t gs){fragment_gemm::run<true>(c,s,x,a,r,f,M,N,K,*reinterpret_cast<nv_bfloat16*>(&gs));}
