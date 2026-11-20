// CUTLASS grouped Hopper R/S mixed-input compile probe based on NVIDIA example 69.
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/device_kernel.h"
using namespace cute;
using Bf16 = cutlass::bfloat16_t;
#ifdef PACKED_FP4
using Quant = cutlass::float_e2m1_t;
#else
using Quant = cutlass::float_e4m3_t;
#endif
using GroupShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using ProbeTile = Shape<_128,_16,Int<PROBE_CTA_K>>;
using Cluster = Shape<_1,_1,_1>;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Epi = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,ProbeTile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,
    float,float,Bf16,Col*,8,Bf16,Col*,8,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
#ifdef PACKED_SCALE
using Scale = uint32_t;
#else
using Scale = Bf16;
#endif
#ifdef SHARED_HUMMING_WEIGHT
#include "humming_atom.hpp"
using WeightLayout = HummingWeightLayout;
#else
using WeightLayout = Row;
#endif
using Main = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    cute::tuple<Quant,Scale>,WeightLayout*,(128 / cutlass::sizeof_bits<Quant>::value),Bf16,Col*,8,float,ProbeTile,Cluster,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<GroupShape,Main,Epi>;
template __global__ void cutlass::device_kernel<Kernel>(CUTLASS_GRID_CONSTANT Kernel::Params const params);
