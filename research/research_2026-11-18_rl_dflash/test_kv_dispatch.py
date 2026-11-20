"""Exercise direct-commit admission using the exact generated worker predicate."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from build_kv import pinned

HERE = Path(__file__).resolve().parent


def predicate():
    # Parse the actual builder string to test the exact source which gets
    # inserted into the worker. The predicate has no dependency on SGLang init.
    builder = ast.parse((HERE / "build_kv_trial.py").read_text(encoding="utf-8-sig"))
    fragment = next(
        n.value
        for n in ast.walk(builder)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and "direct_cache = None" in n.value
    )
    from textwrap import dedent

    plain = dedent(
        fragment[: fragment.index("        self._fused_kv_helper.materialize(")]
    )
    tree = ast.parse(plain)
    check = next(n.test for n in tree.body if isinstance(n, ast.If))
    return compile(
        ast.fix_missing_locations(ast.Expression(check)),
        "generated_kv_dispatch",
        "eval",
    )


class Pool:
    kv_cache_layout = "nhd"
    dtype = store_dtype = torch.bfloat16
    start_layer = 0
    layer_num = 5
    head_num = 1
    head_dim = v_head_dim = 128


class StridedPool(Pool):
    pass


def main():
    code = predicate()
    rows = NS(is_contiguous=lambda: True)
    env = dict(
        torch=torch,
        MHATokenToKVPool=Pool,
        pool=Pool(),
        ctx_hidden=NS(is_cuda=True, dtype=torch.bfloat16),
        commit_lens=rows,
        ctx_cache_loc_2d=rows,
        self=NS(
            _fused_kv_helper=NS(n_layers=5, head_dim=128, num_kv_heads=1),
            draft_model=NS(
                layers=[NS(self_attn=NS(attn=NS(layer_id=i))) for i in range(5)]
            ),
        ),
    )
    cases = {}

    def test(name, expected, **override):
        actual = bool(eval(code, {**env, **override}))
        assert actual == expected, name
        cases[name] = actual

    test("dense_regular_bf16", True)
    test("753_token_prefill", False, commit_lens=None, ctx_cache_loc_2d=None)
    test("16384_token_chunk_prefill", False, commit_lens=None, ctx_cache_loc_2d=None)
    test("pool_subclass", False, pool=StridedPool())
    test("noncontiguous_commit", False, commit_lens=NS(is_contiguous=lambda: False))
    test("cpu", False, ctx_hidden=NS(is_cuda=False, dtype=torch.bfloat16))
    for name, val in [
        ("store_dtype", torch.uint8),
        ("dtype", torch.float16),
        ("kv_cache_layout", "hnd"),
        ("start_layer", 1),
        ("layer_num", 4),
        ("head_dim", 256),
        ("head_num", 8),
    ]:
        pool = Pool()
        setattr(pool, name, val)
        test(name, False, pool=pool)
    (HERE / "kv_dispatch_cpu.json").write_text(json.dumps(cases, indent=2))
    print(len(cases), "direct-cache dispatch fixtures passed.")


if __name__ == "__main__":
    main()
