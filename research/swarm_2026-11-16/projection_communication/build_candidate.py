"""Build an isolated row-sharded projection candidate; no active-source edits."""
from pathlib import Path
import ast, hashlib, json
ROOT = Path('/mnt/d/OpenKernelChallenge/serving')
OUT = ROOT / 'swarm_2026-11-16/projection_communication'
helper_rel = Path('python/sglang/srt/layers/communication/muse_spark_gemm_ag.py')
model_rel = Path('python/sglang/srt/models/muse_spark.py')
helper = (ROOT/'sglang'/helper_rel).read_text()
helper += '''\n\ndef prefill_row_projection_supported(group, x, weight) -> bool:
    """A large replicated projection with equal contiguous token-row shards."""
    if group is None or group.world_size not in (2, 4, 8):
        return False
    if not x.is_cuda or x.ndim != 2 or weight.ndim != 2:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
        return False
    if weight.device != x.device or not weight.is_contiguous():
        return False
    m, k = x.shape
    n = weight.shape[0]
    return (
        torch.cuda.get_device_capability(x.device) == (9, 0)
        and m >= 512
        and m % group.world_size == 0
        and min(k, n) >= 1024
        and k == weight.shape[1]
        and k % 128 == 0
        and n % 128 == 0
    )


def prefill_row_projection(group, x, weight):
    """Compute full output rows once, then gather in original token order.

    Input and weights must be replicated on the TP group. The complete K
    reduction and full output dimension remain local, with FP32 output before
    the single BF16 rounding. No persistent repacked weight or padding exists.
    """
    m_local = x.shape[0] // group.world_size
    start = group.rank_in_group * m_local
    local_fp32 = torch.mm(
        x[start : start + m_local], weight.t(), out_dtype=torch.float32
    )
    local = local_fp32.to(x.dtype)
    del local_fp32
    return group.all_gather(local, dim=0)
'''
model = (ROOT/'research_iteration_2026-11-15/projection_fp32_candidate/candidate_sources'/model_rel).read_text()
model = model.replace('from sglang.srt.layers.communication.muse_spark_prefill_projection import (', 'from sglang.srt.layers.communication.muse_spark_gemm_ag import (')
model = model.replace('prefill_projection_supported', 'prefill_row_projection_supported').replace('prefill_projection', 'prefill_row_projection')
model = model.replace('    reserve_projection_workspace,\n)\nfrom sglang.srt.layers.communication.muse_spark_gemm_ag import (\n    prefill_row_projection,\n    prefill_row_projection_supported,\n)', '    prefill_row_projection,\n    prefill_row_projection_supported,\n    reserve_projection_workspace,\n)')
# The supplied model contains prior router improvements shared with active V4;
# the diff is audited separately before integration by the coordinator.
files = {}
for relative, source in ((helper_rel, helper), (model_rel, model)):
    ast.parse(source)
    target = OUT/'candidate_sources'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    files[str(relative)] = hashlib.sha256(source.encode()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(dict(mechanism='contiguous token-row sharding of replicated prefill projections',files=files,serving_validated=False), indent=2))
print(json.dumps(files, indent=2))
