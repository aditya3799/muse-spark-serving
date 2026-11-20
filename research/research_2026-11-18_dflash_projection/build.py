from difflib import unified_diff
from hashlib import sha256
from pathlib import Path
import ast
import subprocess

base = "297e54fd105e974cdf401e5b4d20253a22c16f90"
root = Path(__file__).resolve().parents[1]
out = Path(__file__).parent
name = "python/sglang/srt/models/dflash.py"
original = subprocess.check_output(["git", "-C", str(root / "sglang"), "show", f"{base}:{name}"]).decode()
parent = (root / "release_2026-11-18/candidate.diff").read_bytes()
assert sha256(parent).hexdigest() == "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
assert f"diff --git a/{name} ".encode() not in parent
new = original
def replace(old, replacement):
    global new
    assert new.count(old) == 1, old
    new = new.replace(old, replacement)

replace("from sglang.srt.utils import is_npu, set_weight_attrs", "from sglang.srt.utils import is_cuda, is_npu, set_weight_attrs")
replace("class DFlashDraftModel(nn.Module):", '''class _ShardedContextProjection(nn.Linear):
    """Project complete output features on each draft TP rank and gather before RMSNorm."""

    def __init__(self, input_size, output_size, group):
        super().__init__(input_size, output_size // group.world_size, bias=False)
        self.tp_group = group
        self.global_output_size = output_size
        if group.rank_in_group == 0:
            logger.info("Sharding DFlash context projection %dx%d across %d ranks.", input_size, output_size, group.world_size)
        set_weight_attrs(
            self.weight, {"weight_loader": sharded_weight_loader(0, parallel_group="tp")}
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Keep a full-K FP32 result until the BF16 projection boundary.
        local = torch.mm(hidden, self.weight.t(), out_dtype=torch.float32).to(hidden.dtype)
        return self.tp_group.all_gather(local, dim=-1)


class DFlashDraftModel(nn.Module):''')
replace("    supports_quantization = False\n", "    supports_quantization = False\n    shard_context_projection = False\n")
replace('''        else:
            self.fc = nn.Linear(
                self.num_context_features * hidden_size, hidden_size, bias=False
            )''','''        elif (
            self.shard_context_projection
            and is_cuda()
            and torch.get_default_dtype() == torch.bfloat16
            and not get_exec().deterministic.enable_deterministic_inference
            and get_parallel().tp_size > 1
            and get_parallel().attn_tp_size == get_parallel().tp_size
            and hidden_size % get_parallel().tp_size == 0
        ):
            self.fc = _ShardedContextProjection(
                self.num_context_features * hidden_size, hidden_size, get_parallel().tp_group
            )
        else:
            self.fc = nn.Linear(
                self.num_context_features * hidden_size, hidden_size, bias=False
            )''')
replace("from sglang.srt.runtime_context import get_parallel, get_spec", "from sglang.srt.runtime_context import get_exec, get_parallel, get_spec")
replace("                    else:\n                        expected_shape = tuple(param.shape)",'''                    elif isinstance(self.fc, _ShardedContextProjection) and resolved_name == "fc.weight":
                        expected_shape = (self.fc.global_output_size, self.fc.in_features)
                        loaded_shape = tuple(loaded_weight.shape)
                        shape_matches = loaded_shape == expected_shape
                    else:
                        expected_shape = tuple(param.shape)''')
replace('''    """DFlash backbone + candidate selector. Reuses the DFLASH speculative worker."""
''','''    """DFlash backbone + candidate selector. Reuses the DFLASH speculative worker."""

    shard_context_projection = True
''')
ast.parse(new)
diff = f"diff --git a/{name} b/{name}\n" + "".join(unified_diff(original.splitlines(True), new.splitlines(True), "a/"+name, "b/"+name))
(out / "dflash.py").write_bytes(new.encode())
(out / "candidate.diff").write_bytes(parent + diff.encode())
print("candidate bytes", len(parent + diff.encode()))

