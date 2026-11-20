#include "muse_spark/all_gather_norm_residual.cuh"
namespace sglang {
template __global__ void muse_spark_all_gather_norm_residual_row_kernel<8192,8,256,false,true,false,true,true>(const __grid_constant__ muse_ag_norm::Params);
template __global__ void muse_spark_all_gather_norm_residual_row_kernel<8192,8,256,false,true,true,true,true>(const __grid_constant__ muse_ag_norm::Params);
template __global__ void muse_spark_all_gather_norm_residual_row_kernel<8192,8,512,false,false,false,true,true>(const __grid_constant__ muse_ag_norm::Params);
template __global__ void muse_spark_all_gather_norm_residual_row_kernel<8192,8,512,false,false,true,true,true>(const __grid_constant__ muse_ag_norm::Params);
}
