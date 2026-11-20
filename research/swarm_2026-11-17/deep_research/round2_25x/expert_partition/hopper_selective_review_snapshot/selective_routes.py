"""One histogram, three exact-disjoint route sets; no host route reads."""
import torch,triton
import triton.language as tl
@triton.jit
def _partition3(ids_ptr,light_ptr,heavy_ptr,m2_ptr,count_ptr,ROUTES:tl.constexpr,EXPERTS:tl.constexpr,CUTOFF:tl.constexpr,HEAVY_BM:tl.constexpr,M2_BM:tl.constexpr,BLOCK:tl.constexpr,BINS:tl.constexpr):
 offsets=tl.arange(0,BLOCK);ids=tl.load(ids_ptr+offsets,mask=offsets<ROUTES,other=-1);valid=(offsets<ROUTES)&(ids>=0)&(ids<EXPERTS)
 bins=tl.where(valid,ids,BINS-1).to(tl.int32);counts=tl.histogram(bins,BINS);route_count=tl.gather(counts,tl.where(valid,ids,0).to(tl.int32),0)
 light=valid&(route_count<=CUTOFF);eligible=valid&(route_count>CUTOFF)&(route_count>16)&(tl.cdiv(route_count,M2_BM)<tl.cdiv(route_count,HEAVY_BM));heavy=valid&(~light)&(~eligible)
 tl.store(light_ptr+offsets,tl.where(light,ids,-1),mask=offsets<ROUTES);tl.store(heavy_ptr+offsets,tl.where(heavy,ids,-1),mask=offsets<ROUTES);tl.store(m2_ptr+offsets,tl.where(eligible,ids,-1),mask=offsets<ROUTES)
 e=tl.arange(0,BINS);tl.store(count_ptr+e,counts,mask=e<EXPERTS)
def partition3(ids,experts,cutoff,heavy_bm=16,m2_bm=32):
 if experts!=32 or cutoff!=16 or heavy_bm!=16 or m2_bm!=32 or ids.shape!=(32,8):raise ValueError('This gate is restricted to the audited native M32 policy')
 if ids.dtype!=torch.int32 or not ids.is_cuda or not ids.is_contiguous():raise ValueError('Expected contiguous CUDA int32 routes')
 outputs=[torch.empty_like(ids) for _ in range(3)];counts=torch.empty((experts,),device=ids.device,dtype=torch.int32)
 _partition3[(1,)](ids,*outputs,counts,ROUTES=ids.numel(),EXPERTS=experts,CUTOFF=cutoff,HEAVY_BM=heavy_bm,M2_BM=m2_bm,BLOCK=triton.next_power_of_2(ids.numel()),BINS=triton.next_power_of_2(experts+1),num_warps=4)
 return (*outputs,counts)
