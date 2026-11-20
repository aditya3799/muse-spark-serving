// Offline-only: exactly one upstream run_flash_fwd specialization per compiler.
#include "flash_fwd_launch_template.h"
#ifndef DIAGNOSTIC_PAGED
#error DIAGNOSTIC_PAGED must be defined
#endif
void instantiate_exact_fa3(Flash_fwd_params& params, cudaStream_t stream) {
    run_flash_fwd<
        90, 64, 64, 1, cutlass::float_e4m3_t, cutlass::bfloat16_t,
        true, false, false, true, DIAGNOSTIC_PAGED,
        false, false, false, true, false, false, false, false
    >(params, stream);
}
