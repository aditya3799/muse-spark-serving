"""Export an unapplied, auditable candidate and check it against the pinned base."""

import argparse
import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

from marlin_bench import SOURCE, EXPECTED_SHA

BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
HELPER = "python/sglang/kernels/ops/moe/adaptive_marlin_tiles.py"


def delta(relative, before, after, new=False):
    header = f"diff --git a/{relative} b/{relative}\n"
    if new:
        header += "new file mode 100644\n"
    return header + "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="/dev/null" if new else "a/" + relative,
            tofile="b/" + relative,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sglang-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compact", action="store_true")
    args = p.parse_args()
    root, out = args.sglang_root, args.out
    out.mkdir(parents=True, exist_ok=False)
    original = (root / SOURCE).read_bytes()
    assert hashlib.sha256(original).hexdigest() == EXPECTED_SHA
    text = original.decode()
    tree = ast.parse(text)
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
    lines = text.splitlines(True)
    method = "".join(lines[fn.lineno - 1 : fn.end_lineno])
    old = "        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm\n"
    assert method.count(old) == 1
    method = method.replace(
        old,
        "        from sglang.kernels.ops.moe.adaptive_marlin_tiles import (\n            adaptive_gemm as moe_wna16_marlin_gemm,\n            prepare_adaptive_alignment,\n        )\n",
    )
    old = "        workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)\n"
    assert method.count(old) == 1
    method = method.replace(
        old,
        "        adaptive_tables = prepare_adaptive_alignment(\n            local_topk_ids, block_size_m, self.num_local_experts,\n            num_tokens_post_padded, expert_map is not None, sorted_token_ids,\n        )\n"
        + old,
    )
    old = "            has_inactive_prefix_count=expert_map is None,\n"
    assert method.count(old) == 2
    method = method.replace(old, old + "            adaptive_tables=adaptive_tables,\n")
    candidate = (
        "".join(lines[: fn.lineno - 1]) + method + "".join(lines[fn.end_lineno :])
    )
    helper_tree = ast.parse(Path(__file__).with_name("adaptive_tiles.py").read_text())
    wanted = (
        {"adaptive_gemm", "prepare_compact_alignment", "_select_and_compact"}
        if args.compact
        else {"adaptive_gemm", "prepare_adaptive_alignment", "_select_tile"}
    )
    nodes = [
        n
        for n in helper_tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        or isinstance(n, ast.FunctionDef)
        and n.name in wanted
    ]
    for node in nodes:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "prepare_compact_alignment"
        ):
            node.name = "prepare_adaptive_alignment"
    helper = (
        '"""Select sparse EP tiles on-device while preserving the dense Marlin path.\n\n'
        'The caller supplies sorted, already-local routing tables with inactive\n'
        'blocks first, caller-owned output buffers, and masked inactive output\n'
        'consumers. The selector preserves every active route and disables the\n'
        'unused projection with a zero padded count; it never reads back to CPU.\n'
        'This experimental helper is used only by the Muse Spark Marlin path.\n'
        '"""\n\n'
        + ast.unparse(ast.Module(body=nodes, type_ignores=[]))
        + "\n"
    )
    ast.parse(candidate)
    ast.parse(helper)
    submitted = json.loads((root.parent / "candidate_manifest.json").read_text())
    for relative, expected in submitted["source_file_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    current = {
        relative: (root / relative).read_text()
        for relative in submitted["changed_files"]
    }
    current[SOURCE] = candidate
    current[HELPER] = helper
    cumulative = ""
    base_dir = out / "base_apply_check"
    base_dir.mkdir()
    for relative, after in current.items():
        if relative == HELPER:
            before = ""
        else:
            before = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{BASE}:{relative}"]
            ).decode()
            base_path = base_dir / relative
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_text(before, encoding="utf-8")
        candidate_path = out / "candidate_sources" / relative
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(after, encoding="utf-8")
        cumulative += delta(relative, before, after, relative == HELPER)
    patch = out / "candidate_unsubmitted.diff"
    patch.write_text(cumulative, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(base_dir), "apply", "--check", str(patch.resolve())],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(base_dir), "apply", str(patch.resolve())], check=True
    )
    for relative, expected in current.items():
        assert (base_dir / relative).read_text() == expected
    added = [
        line[1:]
        for line in cumulative.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert len(patch.read_bytes()) <= 400_000
    assert max(map(len, added)) <= 400
    report = dict(
        status="prepared_unsubmitted",
        policy="adaptive_compact_16" if args.compact else "adaptive_16",
        base_commit=BASE,
        patch_bytes=len(patch.read_bytes()),
        max_added_line_characters=max(map(len, added)),
        patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
        pinned_base_apply_and_source_match=True,
        active_checkout_modified=False,
        serving_validated=False,
        source_file_sha256={relative: hashlib.sha256(after.encode()).hexdigest() for relative, after in current.items()},
        changed_files=list(current),
    )
    (out / "candidate_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
