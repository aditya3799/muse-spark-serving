"""Materialize isolated pinned kernels plus a direct-address fast path."""

import ast
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "sglang"
REV = "297e54fd105e974cdf401e5b4d20253a22c16f90"


def pinned(path):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{REV}:python/sglang/{path}"]
    ).decode()


def block(txt, name):
    node = next(n for n in ast.parse(txt).body if getattr(n, "name", None) == name)
    first = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    return "\n".join(txt.splitlines()[first - 1 : node.end_lineno]) + "\n"


def direct_body(original):
    direct = original.replace("_fused_norm_rope_kernel_stacked(", "_direct_norm_rope(")
    direct = direct.replace(
        "    BLOCK_HD: tl.constexpr,",
        "    BLOCK_HD: tl.constexpr,\n    loc_ptr,\n    commit_ptr,\n    WIDTH: tl.constexpr,",
    )
    start = direct.index("    k_write = (")
    stop = direct.index("    offs = tl.arange", start)
    direct = (
        direct[:start]
        + """    if ctx_id % WIDTH >= tl.load(commit_ptr + ctx_id // WIDTH):
        return
    physical = tl.load(loc_ptr + ctx_id)
    key_storage = tl.load(k_out_ptr + layer_id).to(tl.pointer_type(kv_ptr.dtype.element_ty))
    value_storage = tl.load(v_out_ptr + layer_id).to(tl.pointer_type(kv_ptr.dtype.element_ty))
    k_write = key_storage + physical * k_out_stride_ctx + head_id * k_out_stride_head
    v_write = value_storage + physical * v_out_stride_ctx + head_id * v_out_stride_head

"""
        + direct[stop:]
    )
    return direct


def main():
    f = pinned("kernels/ops/speculative/fused_kv_materialize.py")
    cache = pinned("kernels/ops/kvcache/cache_move.py")
    original = block(f, "_fused_norm_rope_kernel_stacked")
    writer = block(cache, "set_kv_buffer_prefix_valid_tiled")
    direct = direct_body(original)
    text = (
        "import triton\nimport triton.language as tl\n\n"
        + original
        + "\n"
        + writer
        + "\n"
        + direct
    )
    ast.parse(text)
    (HERE / "kv_kernels.py").write_text(text, encoding="utf8")
    print("Pinned arithmetic and writer extracted; direct store source verified.")


if __name__ == "__main__":
    main()
