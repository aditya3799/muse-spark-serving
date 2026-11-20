import ast
import difflib
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).parent / "prefix_scan"
ROOT = Path(__file__).parent.parent.parent
REL = "python/sglang/kernels/ops/attention/metadata.py"
original_bytes = (ROOT / "sglang" / REL).read_bytes()
original = original_bytes.decode("utf-8").replace("\r\n", "\n")
source = original

old = """    # One block computes the sequence lengths and prefix sum.
    if pid_b == 0 and pid_c == 0:
        acc = 0
        for idx in range(B):
            seq = tl.load(seq_lens + idx * seq_lens_stride_0)
            val = (seq + SEQ_LEN_DELTA).to(tl.int32)
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)
            tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc)
            acc += val
        tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc)
"""
new = """    # One block computes the sequence lengths and prefix sum.
    if pid_b == 0 and pid_c == 0:
        if PREFIX_BLOCK > 0:
            rows = tl.arange(0, PREFIX_BLOCK)
            valid = rows < B
            seq = tl.load(seq_lens + rows * seq_lens_stride_0, valid, other=0)
            # Apply the mask after adding delta: padding lanes contribute zero.
            values = tl.where(valid, (seq + SEQ_LEN_DELTA).to(tl.int32), 0)
            inclusive = tl.cumsum(values, axis=0)
            tl.store(
                cache_seqlens_int32 + rows * cache_seqlens_int32_stride_0,
                values,
                valid,
            )
            tl.store(
                cu_seqlens_k + rows * cu_seqlens_k_stride_0,
                inclusive - values,
                valid,
            )
            tl.store(
                cu_seqlens_k + B * cu_seqlens_k_stride_0,
                tl.sum(values, axis=0),
            )
        else:
            # Keep the scalar path for large scans or overlapping output rows.
            acc = 0
            for idx in range(B):
                seq = tl.load(seq_lens + idx * seq_lens_stride_0)
                val = (seq + SEQ_LEN_DELTA).to(tl.int32)
                tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)
                tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc)
                acc += val
            tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc)
"""
assert source.count(old) == 1
source = source.replace(old, new)
signature = "    ZERO_TAIL: tl.constexpr,\n):"
assert source.count(signature) == 1
source = source.replace(signature, "    ZERO_TAIL: tl.constexpr,\n    PREFIX_BLOCK: tl.constexpr,\n):")
call = "        ZERO_TAIL=zero_tail,\n        num_warps=block_cols // 32,"
replacement = """        ZERO_TAIL=zero_tail,
        # Bound scan register use; this depends only on tensor geometry.
        PREFIX_BLOCK=(
            triton.next_power_of_2(max(1, batch_size))
            if batch_size <= 1024
            and cache_seqlens_int32.stride(0) > 0
            and cu_seqlens_k.stride(0) > 0
            else 0
        ),
        num_warps=block_cols // 32,"""
assert source.count(call) == 1
source = source.replace(call, replacement)
ast.parse(source)
candidate = OUT / "candidate" / REL
candidate.parent.mkdir(parents=True, exist_ok=True)
candidate.write_bytes(source.encode("utf-8"))
delta = "".join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
    fromfile="a/" + REL, tofile="b/" + REL))
(OUT / "prefix_delta.diff").write_bytes(delta.encode("utf-8"))
manifest = dict(status="experimental_cpu_only_gpu_unvalidated", encoding="UTF-8 LF",
    original_sha256=hashlib.sha256(original_bytes).hexdigest(),
    original_normalized_sha256=hashlib.sha256(original.encode("utf-8")).hexdigest(),
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    delta_bytes=len(delta.encode("utf-8")), scan_max_batch=1024,
    limitations="No Triton compilation, GPU numerical, CUDA graph or latency validation; independent future option only")
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
