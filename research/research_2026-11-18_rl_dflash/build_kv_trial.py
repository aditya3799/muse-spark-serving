"""Produce a guarded two-file direct-cache trial delta against the scored parent."""

import ast
import difflib
import hashlib
from pathlib import Path
import subprocess

from build_kv import pinned

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NATIVE = "python/sglang/kernels/ops/speculative/fused_kv_materialize.py"
WORKER = "python/sglang/srt/speculative/dflash_worker_v2.py"


def one(s, old, new):
    if s.count(old) != 1:
        raise ValueError("source changed at " + old[:75])
    return s.replace(old, new)


def diff(s, new, path):
    ast.parse(new)
    return "".join(
        difflib.unified_diff(
            s.splitlines(True),
            new.splitlines(True),
            fromfile="a/" + path,
            tofile="b/" + path,
        )
    )


def main():
    parent = (ROOT / "release_2026-11-18/candidate.diff").read_bytes()
    assert (
        hashlib.sha256(parent).hexdigest()
        == "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    )
    assert all(
        ("diff --git a/" + p + " ").encode() not in parent for p in (NATIVE, WORKER)
    )
    original = pinned(NATIVE.removeprefix("python/sglang/"))
    modified = one(
        original,
        "    BLOCK_HD: tl.constexpr,\n):",
        "    BLOCK_HD: tl.constexpr,\n    loc_ptr=None,\n    commit_ptr=None,\n    WIDTH: tl.constexpr = 0,\n):",
    )
    start = modified.index("    k_write = (")
    end = modified.index("    offs = tl.arange", start)
    old_address = modified[start:end].rstrip()
    new_address = """    if WIDTH:
        if ctx_id % WIDTH >= tl.load(commit_ptr + ctx_id // WIDTH):
            return
        physical = tl.load(loc_ptr + ctx_id)
        key_storage = tl.load(k_out_ptr + layer_id).to(tl.pointer_type(kv_ptr.dtype.element_ty))
        value_storage = tl.load(v_out_ptr + layer_id).to(tl.pointer_type(kv_ptr.dtype.element_ty))
        k_write = key_storage + physical * k_out_stride_ctx + head_id * k_out_stride_head
        v_write = value_storage + physical * v_out_stride_ctx + head_id * v_out_stride_head
    else:
"""
    new_address += (
        "\n".join("    " + line for line in old_address.splitlines()) + "\n\n"
    )
    modified = modified[:start] + new_address + modified[end:]
    modified = one(
        modified,
        "        write_layer_kv: Callable[[int, torch.Tensor, torch.Tensor], None],",
        "        write_layer_kv: Callable[[int, torch.Tensor, torch.Tensor], None],\n        direct_cache=None,",
    )
    marker = "        tmp_k = self._k_workspace[:, :total_ctx]"
    modified = one(
        modified,
        marker,
        """        if direct_cache is not None:
            k_ptrs, v_ptrs, loc, lens, row_stride = direct_cache
            _fused_norm_rope_kernel_stacked[(total_ctx, self.num_kv_heads, self.n_layers)](
                proj_out, self.k_norm_weights, self.eps_values, cos_sin_cache, positions,
                k_ptrs, v_ptrs, proj_out.stride(0), proj_out.stride(1),
                self.k_norm_weights.stride(0), cos_sin_cache.stride(0),
                0, row_stride, self.head_dim, 0, row_stride, self.head_dim,
                total_ctx, self.n_layers, self.num_kv_heads, self.head_dim,
                self.num_kv_heads * self.head_dim, self.rotary_dim, self.rotary_dim // 2,
                triton.next_power_of_2(self.head_dim), loc, lens, WIDTH=loc.shape[1],
            )
            return
"""
        + marker,
    )
    method = pinned(WORKER.removeprefix("python/sglang/"))
    start = "        self._fused_kv_helper.materialize("
    method_new = one(
        method,
        start,
        """        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        direct_cache = None
        pool = token_to_kv_pool
        if (
            type(pool) is MHATokenToKVPool
            and pool.kv_cache_layout == "nhd"
            and pool.dtype == pool.store_dtype == ctx_hidden.dtype == torch.bfloat16
            and ctx_hidden.is_cuda
            and ctx_cache_loc_2d is not None and ctx_cache_loc_2d.is_contiguous()
            and commit_lens is not None and commit_lens.is_contiguous()
            and pool.start_layer == 0
            and pool.layer_num == self._fused_kv_helper.n_layers
            and pool.head_dim == pool.v_head_dim == self._fused_kv_helper.head_dim
            and pool.head_num == self._fused_kv_helper.num_kv_heads
            and all(layer.self_attn.attn.layer_id == i for i, layer in enumerate(self.draft_model.layers))
        ):
            direct_cache = (
                pool.k_data_ptrs, pool.v_data_ptrs, ctx_cache_loc_2d,
                commit_lens, pool.head_num * pool.head_dim,
            )
        self._fused_kv_helper.materialize(""",
    )
    method_new = one(
        method_new,
        "            write_layer_kv=_write_layer_kv,",
        "            write_layer_kv=_write_layer_kv,\n            direct_cache=direct_cache,",
    )
    # Benchmark the modified production kernel itself, preserving its inactive
    # constexpr branch and the exact arithmetic from the pinned implementation.
    from build_kv import block

    kernel = block(modified, "_fused_norm_rope_kernel_stacked")
    (HERE / "kv_integrated_kernel.py").write_text(
        "import triton\nimport triton.language as tl\n\n" + kernel, encoding="utf8"
    )
    out = HERE / "kv_trial_parent_delta.diff"
    out.write_bytes(
        (diff(original, modified, NATIVE) + diff(method, method_new, WORKER)).encode()
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT / "sglang"),
            "apply",
            "--check",
            "--whitespace=error",
            str(out),
        ],
        check=True,
    )
    print("Guarded direct-store trial applies against both untouched pinned sources.")


if __name__ == "__main__":
    main()
