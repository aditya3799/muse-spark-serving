from pathlib import Path
import hashlib, json, shutil
here=Path(__file__).resolve().parent
root=here.parents[1]
old=root/'research_2026-11-18_rl_experts/upstream_scale_order'
base=(old/'scale_order.py').read_text()
a=base.index('HEADER_SHA =');z=base.index('\nclass ScaleOrderKernel')
binding=r'''HEADER_SHA = '9a021dffc15f2ba2f3c355b4e508aeb3722b351a8be8130ffd4530743df3edd3'
VARIANT = 'cycles'
@lru_cache(maxsize=4)
def source_for(variant):
    headers = Path(humming.__file__).resolve().parent / 'include/humming'
    kernel = (headers / 'kernel/humming.cuh').read_text()
    if hashlib.sha256(kernel.encode()).hexdigest() != HEADER_SHA:
        return None
    start = kernel.index('    while (slice_iters) {')
    end = kernel.index('\n    consumer.wait_channel();',start)
    text = kernel[start:end]
    if variant in ('tail','cycles_tail'):
        needle='          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);'
        assert text.count(needle)==1
        text=text.replace(needle,'          if (slice_iters > 1 || warp_iter_id + 1 < Ctx::kWarpIters)\n'+needle)
        needle='          mma.transform_b(\n              (warp_iter_id + 1) % 2,\n              (warp_iter_id + 1) % Ctx::kWarpIters);'
        assert text.count(needle)==1
        text=text.replace(needle,'          if (slice_iters > 1 || warp_iter_id + 1 < Ctx::kWarpIters)\n'+needle)
    if variant in ('cycles','cycles_tail'):
        l=text.index('        debug_kernel_timeout_check')
        r=text.index('        slice_iters--;')
        stage=text[l:r].replace('slice_iters','remaining_iters')
        text=('    static_assert(!Ctx::kUseStreamK);\n'
              '    constexpr uint32_t kSliceIters = ProblemShape::K / BlockShape::K;\n'
              '    auto consume_stage = [&](uint32_t stage_id, uint32_t remaining_iters) {\n'+stage+
              '    };\n'
              '    #pragma unroll 1\n'
              '    for (uint32_t cycle = 0; cycle < kSliceIters / kNumStages; ++cycle) {\n'
              '      PRAGMA_UNROLL\n'
              '      for (uint32_t stage = 0; stage < kNumStages; ++stage)\n'
              '        consume_stage(stage, kSliceIters - cycle*kNumStages - stage);\n'
              '    }\n'
              '    PRAGMA_UNROLL\n'
              '    for (uint32_t stage = 0; stage < kSliceIters % kNumStages; ++stage)\n'
              '      consume_stage(stage, kSliceIters % kNumStages - stage);\n')
    if variant not in ('cycles','tail','cycles_tail'):
        raise ValueError(variant)
    text='    static_assert(Ctx::kWarpIters % 2 == 0, "requires alternating operand register banks");\n'+text
    return kernel[:start]+text+kernel[end:]

def adapted_source():
    return source_for(VARIANT)
'''
new=base[:a]+binding+base[z:]
new=new.replace('ScaleOrderKernel','NativeLoopKernel').replace('scale_order_forward','native_loop_forward')
new=new.replace('def kernel_handle(context, meta_json, compute_json, tuning_json):','def kernel_handle(context, meta_json, compute_json, tuning_json, variant):')
new=new.replace('    config=json.loads(meta_json)','    global VARIANT\n    VARIANT=variant\n    config=json.loads(meta_json)')
new=new.replace('json.dumps(kwargs["tuning_config"],sort_keys=True))','json.dumps(kwargs["tuning_config"],sort_keys=True),VARIANT)')
new=new.replace('# Adapted from vLLM Humming PR 107, lesj0610, by selecting probed scale lanes in the pinned reader.\n# https://github.com/vllm-project/humming/pull/107', '# Research backport of fixed-cycle mainloop and live-prefetch bounds from Humming.\n# https://github.com/vllm-project/humming/blob/0832818a89e2de2b3b672ac1ebd076ba18e0a994/humming/include/humming/kernel/humming.cuh')
(here/'native_loop.py').write_text(new)
shutil.copyfile(old/'stage_pair.py',here/'stage_pair.py')
shutil.copyfile(old/'LICENSE.Humming',here/'LICENSE.Humming')
shutil.copyfile(old/'fixture_threshold_cache.json',here/'fixture_threshold_cache.json')
# Rebind only the native GEMM entrypoint in private copies of the frozen serving helpers.
for file,relative in [('muse_spark_nvfp4_moe.py','srt/layers/muse_spark/muse_spark_nvfp4_moe.py'),('muse_spark_humming_split.py','kernels/ops/moe/muse_spark_humming_split.py')]:
 text=(root/'release_2026-11-18/candidate_sources/python/sglang'/relative).read_text()
 text=text.replace('HummingMethod.forward_layer(', 'native_loop_forward(')
 text=text.replace('from humming.layer import HummingMethod','from humming.layer import HummingMethod\n    from native_loop import native_loop_forward')
 (here/file).write_text(text)
print({'bytes':len(new),'sha256':hashlib.sha256(new.encode()).hexdigest()})
