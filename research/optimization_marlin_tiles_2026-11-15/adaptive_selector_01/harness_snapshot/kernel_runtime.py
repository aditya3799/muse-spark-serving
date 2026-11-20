"""Optional source-minimal loader for the isolated real-kernel smoke test.

Skips unrelated serving/package initializers. Uses original function ASTs for
device probes and weight preparation; imports original kernel modules and JIT
compiler. No arithmetic, aligner, GEMM, activation, or finalizer is stubbed.
This is a component test environment, not a full SGLang installation test.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import importlib.machinery
import logging
import os
import sys
import types
from pathlib import Path


def namespace(name, path):
    if name in sys.modules:
        raise RuntimeError(
            f"Source-minimal runtime must start in a fresh process: {name}"
        )
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.origin = module.__file__
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module
    if "." in name:
        parent, attr = name.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], attr, module)
    return module


def selected(name, path, functions, bindings, provenance):
    raw = path.read_bytes()
    tree = ast.parse(raw.decode("utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in functions
    ]
    if {node.name for node in nodes} != set(functions):
        raise RuntimeError(f"Missing requested source definitions: {name}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__dict__.update(bindings)
    module.__spec__ = importlib.util.spec_from_file_location(name, path)
    sys.modules[name] = module
    parent, attr = name.rsplit(".", 1)
    setattr(sys.modules[parent], attr, module)
    body = ast.parse("from __future__ import annotations").body + nodes
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    provenance.append(
        {
            "module": name,
            "file": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "functions": sorted(functions),
            "function_bodies_modified": False,
        }
    )
    return module


def bootstrap(root):
    import torch

    root = Path(root).resolve()
    package = root / "python" / "sglang"
    provenance = []
    parents = (
        "sglang",
        "sglang.srt",
        "sglang.srt.utils",
        "sglang.srt.platforms",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.moe_runner",
        "sglang.srt.layers.moe.moe_runner.triton_utils",
        "sglang.srt.layers.moe.fused_moe_triton",
        "sglang.kernels",
        "sglang.kernels.ops",
    )
    for name in parents:
        namespace(name, package.joinpath(*name.split(".")[1:]))
    cuda_platform = importlib.import_module(
        "sglang.srt.platforms.cuda"
    ).CudaSRTPlatform()
    common = selected(
        "sglang.srt.utils.common",
        package / "srt/utils/common.py",
        (
            "is_cuda",
            "is_hip",
            "is_xpu",
            "is_musa",
            "is_npu",
            "direct_register_custom_op",
        ),
        {
            "torch": torch,
            "lru_cache": functools.lru_cache,
            "sglang_lib": torch.library.Library("sglang", "FRAGMENT"),
            "current_platform": cuda_platform,
        },
        provenance,
    )
    utils = sys.modules["sglang.srt.utils"]
    for name in ("is_cuda", "is_hip", "is_xpu", "is_musa"):
        setattr(utils, name, getattr(common, name))
    # Genuine JIT utilities and kernels; unrelated model/HTTP imports are omitted.
    importlib.import_module("sglang.kernels.jit.utils")
    repack = importlib.import_module(
        "sglang.kernels.ops.quantization.gptq_marlin_repack"
    )
    marlin = selected(
        "sglang.srt.layers.quantization.marlin_utils",
        package / "srt/layers/quantization/marlin_utils.py",
        (
            "marlin_make_workspace",
            "get_scale_perms",
            "marlin_permute_scales",
            "marlin_permute_bias",
        ),
        {"torch": torch},
        provenance,
    )
    selected(
        "sglang.srt.layers.quantization.marlin_utils_fp4",
        package / "srt/layers/quantization/marlin_utils_fp4.py",
        (
            "nvfp4_marlin_process_scales",
            "nvfp4_marlin_process_global_scale",
            "_repack_moe_fp4_weight_for_marlin",
            "_permute_moe_fp4_scales_for_marlin",
            "prepare_moe_nvfp4_layer_for_marlin",
        ),
        {
            "torch": torch,
            "logger": logging.getLogger("tile_experiment"),
            "gptq_marlin_repack": repack.gptq_marlin_repack,
            "marlin_make_workspace": marlin.marlin_make_workspace,
            "marlin_permute_scales": marlin.marlin_permute_scales,
            "marlin_permute_bias": marlin.marlin_permute_bias,
        },
        provenance,
    )
    alignment = importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"
    )
    sys.modules[
        "sglang.srt.layers.moe.fused_moe_triton"
    ].moe_align_block_size = alignment.moe_align_block_size
    return {
        "mode": "source_minimal",
        "namespace_initializers_omitted": list(parents),
        "selected_original_definitions": provenance,
        "arithmetic_stubs": False,
        "full_serving_import_test": False,
        "cuda_home": os.environ.get("CUDA_HOME"),
    }


def main():
    """Run the unchanged benchmark driver after preparing imports."""
    import argparse
    import json
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        report = bootstrap(args.sglang_root)
        (args.out / "runtime_provenance.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        import marlin_bench

        sys.argv = [
            str(Path(marlin_bench.__file__)),
            "--sglang-root",
            str(args.sglang_root),
            "--out",
            str(args.out),
        ] + remaining
        return marlin_bench.main()
    except Exception:
        failure = {
            "status": "failed",
            "stage": "source_minimal_bootstrap",
            "traceback": traceback.format_exc(),
        }
        (args.out / "runtime_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
