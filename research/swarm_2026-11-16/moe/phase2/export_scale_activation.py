"""Export isolated production-path exact-scale fusion, without promoting it."""
from pathlib import Path
import ast,hashlib,json,re,shutil,difflib
P=Path(__file__).resolve().parent
BASE=P.parents[1]/'coordination/round5_humming_adaptive_fused/candidate_sources'
OUT=P/'scale_activation_export';DEST=OUT/'candidate_sources'
model='python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
split='python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
wrapper='python/sglang/kernels/ops/moe/muse_spark_scale_activation.py'
header='python/sglang/kernels/jit/csrc/elementwise/muse_spark_scale_activation.cuh'
pattern=r'(?m)^    scale_active_rows\(cache1,\s*local_ids,\s*layer.w13_post_scale\)\n(    cache2 = [^\n]+)\n    silu_and_mul\(cache1,\s*cache2,\s*expert_ids=local_ids.flatten\(\),\s*expert_step=1\)'
for rel in (model,split):
    original=(BASE/rel).read_text()
    changed,count=re.subn(pattern,r'\1\n    from sglang.kernels.ops.moe.muse_spark_scale_activation import scaled_silu_and_mul\n    scaled_silu_and_mul(cache1,cache2,local_ids,layer.w13_post_scale)',original)
    assert count==1,(rel,count)
    ast.parse(changed);path=DEST/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(changed)
code='''"""Preserve the BF16 global-scale boundary while fusing it with stock SiLU."""
import torch
from sglang.kernels.jit.utils import (
    cache_once, get_activation_cuda_cflags, is_arch_support_pdl, load_jit, make_cpp_args,
)
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH


@cache_once
def _scaled_activation_module():
    args = make_cpp_args(torch.bfloat16, is_arch_support_pdl())
    return load_jit(
        "muse_spark_scale_activation", *args,
        cuda_files=["elementwise/muse_spark_scale_activation.cuh"],
        extra_cuda_cflags=get_activation_cuda_cflags(),
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
        cuda_wrappers=[("run", f"ScaledActivationKernel<{args}>::run")],
    )


def _supports_scaled_activation(input, output, local_ids, scales):
    if torch.version.hip is not None or torch.compiler.is_compiling():
        return False
    if input.ndim != 2 or output.ndim != 2 or scales.ndim != 1:
        return False
    if any(t.dtype != torch.bfloat16 for t in (input, output, scales)):
        return False
    if local_ids.dtype not in (torch.int32, torch.int64):
        return False
    if not input.is_cuda or any(t.device != input.device for t in (output, local_ids, scales)):
        return False
    if not all(t.is_contiguous() for t in (input, output, local_ids, scales)):
        return False
    width = output.shape[1]
    if width <= 0 or width % 8 or input.shape != (output.shape[0], 2 * width):
        return False
    if local_ids.numel() != input.shape[0] or scales.numel() == 0:
        return False
    if input.shape[0] * (width // 8) > (2**32 - 1) // 2:
        return False
    return torch.cuda.get_device_capability(input.device)[0] >= 8


def scaled_silu_and_mul(input, output, local_ids, scales):
    """Internal mapped-route helper; IDs are -1 or valid local expert indices.

    The input/output buffers are disjoint. Scaled gate/up values are consumed
    only by SiLU, so the fused path does not materialize them back into input.
    """
    if not _supports_scaled_activation(input, output, local_ids, scales):
        from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
        from sglang.kernels.ops.activation.activation import silu_and_mul
        scale_active_rows(input, local_ids, scales)
        return silu_and_mul(input, output, expert_ids=local_ids.flatten(), expert_step=1)
    if input.shape[0] == 0:
        return output
    ids = local_ids.reshape(-1).to(torch.int32)
    _scaled_activation_module().run(input, output, ids, scales)
    return output
'''
ast.parse(code);(DEST/wrapper).parent.mkdir(parents=True,exist_ok=True);(DEST/wrapper).write_text(code)
(DEST/header).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/'scale_activation.cuh',DEST/header)
files=(model,split,wrapper,header)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(DEST/header)=='7a107cd80006c384f104294c7b835af0b9349b4fcd3c4cc9f52541e9c7304389'
manifest={'status':'EXPERIMENTAL_SOURCE_EXPORT_NOT_PROMOTED_NOT_SUBMITTED','baseline_job':'e399c79bb1994630','baseline_score':1.2237,
 'sources':{rel:sha(DEST/rel) for rel in files},'baseline_source_hashes':{rel:sha(BASE/rel) for rel in (model,split)},
 'precision':'Exact BF16 scale rounding then pinned stock CUDA SiLU; unchanged GEMM configs/partitions/finalizer',
 'validation':['Pair92 bitwise checks','Fullchain90 partial-run plus68 remaining-run comparisons all bitwise equal; all20 planned M1/M8/M33/M128 pattern cases covered','SM90a PDLtrue exact CUDA header compile passed; no H200 runtime proof'],
 'performance':'Completed local active fullchain ratios near parity1.0013..1.0193; partial run timed out. Pair absolute timings suspicious. No promotion or H200 gain claim.',
 'fallback':'Original scale_active_rows then silu_and_mul retained for unsupported dtype/shape/layout/device/architecture, HIP or torch.compile',
 'scope':'New wrapper path and metadata guards require exact-export validation before integration; native guard/static checks alone are not GPU runtime proof',
 'extra_persistent_weights_bytes':0,'extra_activation_scratch_bytes':0}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
patch=[]
for rel in files:
    old=(BASE/rel).read_text().splitlines(True) if (BASE/rel).exists() else []
    new=(DEST/rel).read_text().splitlines(True)
    patch.extend(difflib.unified_diff(old,new,fromfile=('a/'+rel if old else '/dev/null'),tofile='b/'+rel))
(OUT/'review_delta.diff').write_text(''.join(patch))
print(json.dumps(manifest,indent=2))