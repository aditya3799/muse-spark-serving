// Isolated swap-AB MMA adapter. Original cp.async/scheduler/arithmetic/K reducer remain pinned.
template<class Ctx,class ArithClass> struct ResearchSwapMMA {
 using MmaOpClass=typename Ctx::MmaOpClass;
 using WarpShape=typename Ctx::WarpShape;
 using BlockShape=typename Ctx::BlockShape;
 using SharedStorage=typename Ctx::SharedStorage;
 using ElementA=typename Ctx::ElementA;
 using ElementB=typename Ctx::ElementB;
 using CRegistersType=typename MmaOpClass::CRegisters;
 using CRegistersArrayType=CRegistersType[WarpShape::N/16][WarpShape::M/8];
 static_assert(!Ctx::kUseWgmma && !Ctx::kUseWarpSpec && !Ctx::kUseTmaA && !Ctx::kUseTmaB && !Ctx::kUseTmaC);
 static_assert(Ctx::kIsIndexedGemm && Ctx::kUseCpAsync && !Ctx::kUseStreamK);
 static_assert(WarpShape::N==64 && BlockShape::M==WarpShape::M);
 static_assert((BlockShape::K==64 && (WarpShape::K==32 || WarpShape::K==64)) || (BlockShape::K==128 && WarpShape::K==64));
 static_assert(WarpShape::M==8 || WarpShape::M==16 || WarpShape::M==24);
 static_assert(ElementA::kBits==16 && ElementB::kBits==4 && Ctx::kWeightScaleGroupSize==16);
 static_assert(!Ctx::kHasZeroPoint && !Ctx::kHasBias && Ctx::kInputScaleGroupSize==0);
 Ctx &ctx;ArithClass &arith;
 uint32_t regs_qb[2][4];
 uint32_t regs_weight_a[2][WarpShape::N/16][4];
 uint32_t regs_act_b[2][WarpShape::M/8][2];
 CRegistersArrayType regs_c;
 CUDA_INLINE ResearchSwapMMA(Ctx &c,ArithClass&a):ctx(c),arith(a){}
 CUDA_INLINE void zero_accum(){
  uint32_t*p=reinterpret_cast<uint32_t*>(regs_c);
  PRAGMA_UNROLL
  for(uint32_t i=0;i<sizeof(regs_c)/4;i++)p[i]=0;
 }
 CUDA_INLINE void transform_b(uint32_t buffer,uint32_t iter){
  uint32_t standard_qb[4];
  PRAGMA_UNROLL
  for(uint32_t i=0;i<4;i++){
   uint32_t w=regs_qb[buffer][i];
   standard_qb[i]=(w&0xf00ff00fu)|((w&0x00f000f0u)<<4)|((w&0x0f000f00u)>>4);
  }
  PRAGMA_UNROLL
  for(uint32_t i=0;i<WarpShape::N/16;i++){
   uint32_t b[4];uint32_t zp[4]={0,0,0,0};
   if constexpr(@DIRECT@){
    dequant<ElementB,ElementA,false,false,1>(regs_qb[buffer],b,i,zp);
    arith.may_apply_bs_and_zp_on_b(b,i,buffer);
    PRAGMA_UNROLL
    for(uint32_t r=0;r<4;r++)regs_weight_a[buffer][i][r]=b[r];
   }else{
    dequant<ElementB,ElementA,false,false,1>(standard_qb,b,i,zp);
    arith.may_apply_bs_and_zp_on_b(b,i,buffer);
    regs_weight_a[buffer][i][0]=b[0];regs_weight_a[buffer][i][1]=b[2];
    regs_weight_a[buffer][i][2]=b[1];regs_weight_a[buffer][i][3]=b[3];
   }
  }
 }
 CUDA_INLINE void load_activation(const int4*smem_ptr,uint32_t buffer,uint32_t iter,uint32_t stage){
  uint32_t base=(offsetof(SharedStorage,stages)+stage*sizeof(typename SharedStorage::StageStorage))/128%8;
  PRAGMA_UNROLL
  for(uint32_t t=0;t<WarpShape::M/8;t++){
   PRAGMA_UNROLL
   for(uint32_t reg=0;reg<2;reg++){
    uint32_t row=ctx.m_warp_offset()+t*8+ctx.lane_id()/4;
    uint32_t col=iter*2+ctx.k_warp_offset()/8+reg;
    row=BlockShape::M*(col/8)+row;
    col=(col%8)^((row+base)%8);
    const uint32_t*source=reinterpret_cast<const uint32_t*>(smem_ptr+row*8+col);
    regs_act_b[buffer][t][reg]=source[ctx.lane_id()%4];
   }
  }
 }
 CUDA_INLINE void run(uint32_t stage,uint32_t iter){
  uint32_t buffer=iter%2;
  PRAGMA_UNROLL
  for(uint32_t t=0;t<WarpShape::M/8;t++){
   if constexpr(@SEMISTATIC@){
    if(t>0 && ctx.smem.wr_row_index[ctx.m_warp_offset()+t*8]>=ctx.params.shape_m*ctx.params.top_k)continue;
   }
   PRAGMA_UNROLL
   for(uint32_t n=0;n<WarpShape::N/16;n++)MmaOpClass::fma(regs_weight_a[buffer][n],regs_act_b[buffer][t],regs_c[n][t],regs_c[n][t]);
  }
 }
 template<class T=uint32_t>CUDA_INLINE T*regs_qb_as_ptr(uint32_t b){return reinterpret_cast<T*>(regs_qb[b]);}
 template<class T=uint32_t>CUDA_INLINE T*regs_c_as_ptr(uint32_t b=0){return reinterpret_cast<T*>(regs_c);}
 template<class T=uint32_t>CUDA_INLINE T*final_regs_c_as_ptr(){return reinterpret_cast<T*>(regs_c);}
};
