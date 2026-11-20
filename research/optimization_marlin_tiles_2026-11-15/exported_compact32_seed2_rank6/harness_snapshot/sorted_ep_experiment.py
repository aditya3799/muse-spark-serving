"""Build/load an explicit opt-in sorted-local-EP experiment outside the checkout.

EP remains true. The original generic EP path remains the default. Only callers
certifying ascending *local* expert IDs may opt in; no post-sort remap is allowed.
This first experiment does not enable the inactive-prefix-count specialization.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Source contract changed: {old[:100]}")
    return text.replace(old, new)


def build(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    relative = "python/sglang/kernels/jit/csrc/gemm"
    for name in ("marlin", "marlin_moe"):
        source_dir = root / relative / name
        target_dir = destination / "gemm" / name
        if target_dir.exists():
            # Reuse only a completely known overlay, never silently overwrite it.
            raise RuntimeError(f"Overlay already exists: {target_dir}")
        shutil.copytree(source_dir, target_dir)
    cu_relative = relative + "/marlin_moe/moe_wna16_marlin.cuh"
    original_cu = (root / cu_relative).read_text(encoding="utf-8")
    candidate_cu = replace_once(
        original_cu,
        '  static_assert(!kIsEP || !kExpertIdsAreSorted, "Sorted block IDs are specialized only without EP");',
        "  // Sorted EP is valid for caller-certified local IDs with no later remapping.\n"
        "  // Preserve kIsEP=true and the runtime specialization consistency check below.",
    )
    overlay_cu = destination / "gemm/marlin_moe/moe_wna16_marlin.cuh"
    overlay_cu.write_text(candidate_cu, encoding="utf-8")
    py_relative = "python/sglang/kernels/ops/moe/moe_wna16_marlin.py"
    original_py = (root / py_relative).read_text(encoding="utf-8")
    candidate_py = replace_once(
        original_py,
        "    has_inactive_prefix_count: bool = False,\n) -> torch.Tensor:",
        "    has_inactive_prefix_count: bool = False,\n    allow_sorted_ep: bool = False,\n) -> torch.Tensor:",
    )
    candidate_py = replace_once(
        candidate_py,
        "    The sorted specialization is only used without expert parallelism.",
        "    The sorted specialization is used without EP by default. With EP,\n"
        "    allow_sorted_ep additionally certifies already-local ascending IDs,\n"
        "    inactive blocks first, and no subsequent expert remapping.\n"
        "    Uncertified EP callers keep the generic path.",
    )
    candidate_py = replace_once(
        candidate_py,
        "    device = a.device\n",
        "    if allow_sorted_ep and not (is_ep and expert_ids_are_sorted):\n"
        "        raise ValueError('Sorted EP requires explicit EP and sorted-local-ID contracts')\n"
        "    device = a.device\n",
    )
    candidate_py = replace_once(
        candidate_py,
        "        expert_ids_are_sorted and not is_ep,",
        "        expert_ids_are_sorted and (not is_ep or allow_sorted_ep),",
    )
    patch = "".join(
        difflib.unified_diff(
            original_py.splitlines(True),
            candidate_py.splitlines(True),
            fromfile="a/" + py_relative,
            tofile="b/" + py_relative,
        )
    )
    patch += "".join(
        difflib.unified_diff(
            original_cu.splitlines(True),
            candidate_cu.splitlines(True),
            fromfile="a/" + cu_relative,
            tofile="b/" + cu_relative,
        )
    )
    (destination / "SORTED_EP_API_DELTA.diff").write_text(patch, encoding="utf-8")
    # Runtime-only absolute path directs compilation to the isolated overlay.
    runtime_py = replace_once(
        candidate_py,
        'cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],',
        f"cuda_files=[{str(overlay_cu.resolve())!r}],",
    )
    runtime_path = destination / "sorted_marlin_runtime.py"
    runtime_path.write_text(runtime_py, encoding="utf-8")
    report = {
        "kind": "sorted-local EP opt-in experiment",
        "is_ep_preserved": True,
        "inactive_prefix_count_under_ep": False,
        "active_checkout_modified": False,
        "source_sha256": {
            py_relative: hashlib.sha256((root / py_relative).read_bytes()).hexdigest(),
            cu_relative: hashlib.sha256((root / cu_relative).read_bytes()).hexdigest(),
        },
        "runtime_path": str(runtime_path.resolve()),
    }
    (destination / "overlay_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return runtime_path


def install_dispatcher(root, destination):
    """Route opt-in calls to the overlay; stock calls use the original function."""
    import sglang.kernels.ops.moe.moe_wna16_marlin as stock

    root, destination = Path(root), Path(destination)
    sources = [
        root / "python/sglang/kernels/ops/moe/moe_wna16_marlin.py",
        Path(__file__),
    ]
    for part in ("marlin", "marlin_moe"):
        sources.extend(
            sorted((root / "python/sglang/kernels/jit/csrc/gemm" / part).rglob("*"))
        )
    digest = hashlib.sha256()
    for source in sources:
        if source.is_file():
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    shared = destination.parent.parent / "sorted_ep_builds" / digest.hexdigest()[:20]
    integrity = shared / "overlay_checksums.json"
    if not shared.exists():
        runtime_path = build(root, shared)
        checksums = {
            str(p.relative_to(shared)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in shared.rglob("*")
            if p.is_file()
        }
        integrity.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    else:
        checksums = json.loads(integrity.read_text(encoding="utf-8"))
        for relative, expected in checksums.items():
            if hashlib.sha256((shared / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Cached overlay changed: {relative}")
        runtime_path = shared / "sorted_marlin_runtime.py"
    destination.mkdir(parents=True, exist_ok=False)
    for name in (
        "SORTED_EP_API_DELTA.diff",
        "overlay_manifest.json",
        "overlay_checksums.json",
    ):
        shutil.copy2(shared / name, destination / name)
    spec = importlib.util.spec_from_file_location(
        "tile_sorted_marlin_runtime", runtime_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = stock.moe_wna16_marlin_gemm

    def dispatch(*args, allow_sorted_ep=False, **kwargs):
        if allow_sorted_ep:
            return module.moe_wna16_marlin_gemm(*args, allow_sorted_ep=True, **kwargs)
        return original(*args, **kwargs)

    stock.moe_wna16_marlin_gemm = dispatch
    return {
        "overlay": str(shared),
        "record": str(destination),
        "stock_default_preserved": True,
        "ep_flag_falsified": False,
    }
