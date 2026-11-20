"""Execute the actual exported candidate method and helper in the real benchmark."""

import argparse
import ast
import hashlib
import importlib
import json
import sys
from pathlib import Path

from kernel_runtime import bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sglang-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    args, remaining = p.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runtime = bootstrap(args.sglang_root)
    import torch
    import marlin_bench
    from tile_study import BLOCKS

    candidate_manifest = json.loads(
        (args.candidate / "candidate_manifest.json").read_text()
    )
    policy = candidate_manifest["policy"]
    path = args.candidate / "candidate_sources" / marlin_bench.SOURCE
    source = path.read_bytes()
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MuseSpark1xFp4Experts"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_marlin_expert_outputs"
    )
    namespace = {"torch": torch, "_MARLIN_BLOCK_SIZE_M_CANDIDATES": BLOCKS}
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + [fn],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    helper_dir = args.candidate / "candidate_sources/python/sglang/kernels/ops/moe"
    package = importlib.import_module("sglang.kernels.ops.moe")
    package.__path__.insert(0, str(helper_dir))
    helper = importlib.import_module("sglang.kernels.ops.moe.adaptive_marlin_tiles")
    assert Path(helper.__file__).resolve().is_relative_to(helper_dir.resolve())
    original_loader = marlin_bench.load_method

    def load(root, requested, torch_module=None):
        if requested == policy:
            return namespace["_marlin_expert_outputs"], ast.unparse(fn)
        return original_loader(root, requested, torch_module)

    marlin_bench.load_method = load
    evidence = dict(
        runtime=runtime,
        candidate=str(args.candidate),
        policy=policy,
        method_source_sha256=hashlib.sha256(source).hexdigest(),
        helper_source_sha256=hashlib.sha256(
            Path(helper.__file__).read_bytes()
        ).hexdigest(),
        exported_method_body_modified=False,
        arithmetic_stubs=False,
    )
    (args.out / "exported_candidate_provenance.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    sys.argv = [
        str(Path(marlin_bench.__file__)),
        "--sglang-root",
        str(args.sglang_root),
        "--out",
        str(args.out),
    ] + remaining
    return marlin_bench.main()


if __name__ == "__main__":
    raise SystemExit(main())
