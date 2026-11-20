"""Compile the actual tree selection and sliding-window metadata programs for both GPUs."""
import hashlib
import json
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from fused_tree import _grow
from tree_pages import _tree_suffix
from tree_expand import _expand
from tree_commit import _commit

grow = dict(ids="*i64", scores="*fp32", log_mass="*fp32", anchor="*i64",
            out_ids="*i64", out_parents="*i32", out_pos="*i32", out_mask="*u32",
            inv_temp="fp32", log_fork="fp32")
suffix = dict(dst="*i32", prefix="*i32", branches="*i32", lens="*i32", anc="*i32",
              mapping="*i64", row_stride="i32", prefix_stride="i32", branch_stride="i32",
              window="i32")
expand = dict(slots="*i32", masks="*u32", depths="*i32", seq_len="*i32",
              page_table="*i32", expand_len="*i32", swa_len="*i32", positions="*i64")
commit = dict(nodes="*i64", parents="*i32", predictions="*i64", slots="*i32",
              req_ids="*i64", prefix_lens="*i32", remaining="*i32", request_table="*i32",
              emitted="*i64", counts="*i32", bonus="*i64", src_at_dest="*i32",
              new_lens="*i32", table_stride="i32", live="i32")
rows = []
for sm in (86, 90):
    for width in (8, 16, 32):
        for kernel, arguments, constants in (
            (_grow, grow, dict(DEPTH=7, K=16, N=width, ROWS=width, MASS=True)),
            (_tree_suffix, suffix, dict(NODES=width, BLOCK=256)),
            (_expand, expand, dict(N=width)),
            (_commit, commit, dict(WIDTH=width)),
        ):
            signature = dict(arguments, **{key: "constexpr" for key in constants})
            attrs = {(kernel.arg_names.index(k),): [["tt.divisibility", 16]]
                     for k, v in arguments.items() if v.startswith("*")}
            source = ASTSource(kernel, signature, constants, attrs)
            obj = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                                 options=dict(num_warps=4, num_stages=2))
            binary = obj.asm["cubin"]
            row = dict(sm=sm, nodes=width, name=kernel.__name__, cubin_bytes=len(binary),
                       shared_bytes=obj.metadata.shared, sha256=hashlib.sha256(binary).hexdigest())
            rows.append(row)
            print(sm, width, kernel.__name__, row["cubin_bytes"], row["shared_bytes"], flush=True)
assert not torch.cuda.is_initialized()
Path(__file__).with_name("offline_tree.json").write_text(json.dumps(rows, indent=2) + "\n")
