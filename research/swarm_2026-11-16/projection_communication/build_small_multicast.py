from pathlib import Path
import ast,hashlib,json
P=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/projection_communication')
root=P/'candidate_sources'
out=P/'candidate_small_multicast'
hrel=Path('python/sglang/srt/layers/communication/muse_spark_gemm_ag.py')
mrel=Path('python/sglang/srt/models/muse_spark.py')
h=(root/hrel).read_text()
h+='''\n\ndef small_projection_supported(group, x, weight) -> bool:
    """Hopper column shards through an already-reserved multicast workspace."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if group is None or group.world_size not in (2, 4, 8):
        return False
    comm = group.ca_comm
    if (
        not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or not comm.has_multicast
        or comm.world_size != group.world_size
    ):
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
        and 1 <= m < 512
        and min(k, n) >= 1024
        and k == weight.shape[1]
        and k % 128 == 0
        and n % (128 * group.world_size) == 0
        and m * n * 2 <= comm.max_push_size
        and comm.config.num_push_blocks >= 2
    )


def small_projection_local(comm, x, weight):
    """Complete K reductions for this rank's contiguous output columns."""
    n_local = weight.shape[0] // comm.world_size
    start = comm.rank * n_local
    local_fp32 = torch.mm(
        x, weight[start : start + n_local].t(), out_dtype=torch.float32
    )
    return local_fp32.to(x.dtype)


def small_projection(group, x, weight):
    """Gather complete BF16 columns without an intermediate layout copy."""
    comm = group.ca_comm
    local = small_projection_local(comm, x, weight)
    output = torch.empty(
        x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device
    )
    return all_gather(comm, local, output)
'''
m=(root/mrel).read_text()
m=m.replace('    reserve_projection_workspace,\n)', '    reserve_projection_workspace,\n    small_projection,\n    small_projection_supported,\n)',1)
needle = '        self.experts.use_megamoe = self.use_megamoe\n'
replace = needle + '        if (\n            self.pre_expert_proj is not None\n            and self.tp_size in (4, 8)\n            and getattr(self.experts, "use_marlin", False)\n            and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()\n            and parallel.attn_dp_size == 1\n            and parallel.attn_cp_size == 1\n            and torch.cuda.is_available()\n            and torch.cuda.get_device_capability() == (9, 0)\n        ):\n            # This shared group workspace grows only on the first layer.\n            # Reserve before any forward, workspace binding or graph capture.\n            reserve_projection_workspace(\n                parallel.tp_group.ca_comm,\n                max(self.expert_hidden_size, self.hidden_size),\n            )\n'
assert m.count(needle) == 1
m = m.replace(needle, replace)
needle='''            output = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)'''
replace='''            if self._can_shard_small_projection(output, self.post_expert_proj.weight):
                return small_projection(
                    get_parallel().tp_group, output, self.post_expert_proj.weight
                )
'''+needle
assert m.count(needle)==1
m=m.replace(needle,replace)
needle='''    def _defers_unpool_gather(self, comm, num_tokens: int) -> bool:'''
replace='''    def _can_shard_small_projection(self, x, weight) -> bool:
        parallel = get_parallel()
        return (
            getattr(self.experts, "use_marlin", False)
            and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
            and not getattr(get_forward(), "sp_active", False)
            and parallel.attn_dp_size == 1
            and parallel.attn_cp_size == 1
            and small_projection_supported(parallel.tp_group, x, weight)
        )

'''+needle
assert m.count(needle)==1
m=m.replace(needle,replace)
needle='''            else:
                expert_input = bf16_gemm_dispatch(
                    hidden_states, self.pre_expert_proj.weight, None
                )'''
replace='''            elif self._can_shard_small_projection(
                hidden_states, self.pre_expert_proj.weight
            ):
                expert_input = small_projection(
                    get_parallel().tp_group, hidden_states, self.pre_expert_proj.weight
                )
'''+needle
assert m.count(needle)==1
m=m.replace(needle,replace)
files={}
for rel,s in ((hrel,h),(mrel,m)):
 ast.parse(s)
 target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
 files[str(rel)]=hashlib.sha256(s.encode()).hexdigest()
(P/'small_multicast_manifest.json').write_text(json.dumps(dict(source_hashes=files,status='static_only_gpu_and_multicast_unvalidated',preserves_row_candidate=True),indent=2))
print(json.dumps(files,indent=2))
