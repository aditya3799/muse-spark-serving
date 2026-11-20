from pathlib import Path
import ast, difflib, hashlib, json, subprocess

root = Path('/mnt/d/OpenKernelChallenge/serving')
repo = root / 'sglang'
out = root / 'research_iteration_2026-11-15/projection_candidate'
source_root = out / 'candidate_sources'
previous = root / 'optimization_marlin_tiles_2026-11-15/candidate_compact32_01/candidate_sources'
base = '297e54fd105e974cdf401e5b4d20253a22c16f90'
files = {str(p.relative_to(previous)): p.read_text() for p in previous.rglob('*.py')}
helper_path = 'python/sglang/srt/layers/communication/muse_spark_prefill_projection.py'
files[helper_path] = '''"""Shard replicated BF16 prefill projections without splitting reductions.

The caller must supply the same input and full weight on every TP rank.
Each rank computes complete output columns, then gathers them in rank order.
Weights remain in their original layout; no persistent repack is retained.
"""
from __future__ import annotations

import torch


def prefill_projection_supported(group, x, weight) -> bool:
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
        and min(k, n) >= 1024
        and k == weight.shape[1]
        and k % 128 == 0
        and n % (128 * group.world_size) == 0
    )


def prefill_projection(group, x, weight):
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    n_local = weight.shape[0] // group.world_size
    start = group.rank_in_group * n_local
    local = bf16_gemm_dispatch(x, weight[start : start + n_local], None)
    return group.all_gather(local, dim=-1)
'''
model_path = 'python/sglang/srt/models/muse_spark.py'
model = files[model_path]
def replace(old, new):
    global model
    assert model.count(old) == 1, old
    model = model.replace(old, new)
replace('from sglang.srt.layers.cp.utils import is_cp_active', '''from sglang.srt.layers.communication.muse_spark_prefill_projection import (
    prefill_projection,
    prefill_projection_supported,
)
from sglang.srt.layers.cp.utils import is_cp_active''')
replace('    def _defers_unpool_gather(self, comm, num_tokens: int) -> bool:', '''    def _can_shard_prefill_projection(self, x, weight) -> bool:
        parallel = get_parallel()
        return (
            getattr(self.experts, "use_marlin", False)
            and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
            and not getattr(get_forward(), "sp_active", False)
            and parallel.attn_dp_size == 1
            and parallel.attn_cp_size == 1
            and prefill_projection_supported(parallel.tp_group, x, weight)
        )

    def _defers_unpool_gather(self, comm, num_tokens: int) -> bool:''')
replace('''            output = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)''', '''            if self._can_shard_prefill_projection(output, self.post_expert_proj.weight):
                return prefill_projection(
                    get_parallel().tp_group, output, self.post_expert_proj.weight
                )
            output = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)''')
replace('''            else:
                expert_input = bf16_gemm_dispatch(
                    hidden_states, self.pre_expert_proj.weight, None
                )''', '''            elif self._can_shard_prefill_projection(
                hidden_states, self.pre_expert_proj.weight
            ):
                expert_input = prefill_projection(
                    get_parallel().tp_group, hidden_states, self.pre_expert_proj.weight
                )
            else:
                expert_input = bf16_gemm_dispatch(
                    hidden_states, self.pre_expert_proj.weight, None
                )''')
files[model_path] = model
patch = ''
manifest = {}
for path, content in files.items():
    ast.parse(content)
    dest = source_root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    old = subprocess.run(['git', 'show', f'{base}:{path}'], cwd=repo,
                         text=True, capture_output=True)
    assert old.returncode == 0 or not (repo/path).exists(), old.stderr
    patch += ''.join(difflib.unified_diff(
        old.stdout.splitlines(True), content.splitlines(True),
        fromfile='a/'+path if old.returncode == 0 else '/dev/null', tofile='b/'+path))
    manifest[path] = hashlib.sha256(content.encode()).hexdigest()
dest = out / 'candidate.diff'
dest.write_text(patch)
assert len(patch.encode()) < 400000
assert max(len(s) for s in patch.splitlines() if s.startswith('+')) <= 401
(out / 'manifest.json').write_text(json.dumps(dict(base=base, files=manifest,
    patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
    status='experimental; not submitted; H200 validation required',
    parent_job='468701af35b84e07', active_checkout_modified=False), indent=2))
print(json.dumps(dict(files=len(files), bytes=len(patch.encode()), patch=str(dest))))
