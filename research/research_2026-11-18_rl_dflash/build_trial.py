"""Build the isolated trial patch on the exact measured parent without modifying SGLang."""

import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PARENT_SHA = "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
ROOT = HERE.parent
REPO = ROOT / "sglang"
WORKER = "python/sglang/srt/speculative/dflash_worker_v2.py"
MODEL = "python/sglang/srt/models/dflash.py"


def show(name):
    return subprocess.check_output(
        ["git", "-C", str(REPO), "show", BASE + ":" + name]
    ).decode()


def patch(name, original, changed):
    ast.parse(changed)
    return "".join(
        difflib.unified_diff(
            original.splitlines(True),
            changed.splitlines(True),
            fromfile="a/" + name,
            tofile="b/" + name,
        )
    )


def main():
    parent = (ROOT / "release_2026-11-18/candidate.diff").read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT_SHA
    for name in (WORKER, MODEL):
        assert ("diff --git a/" + name + " ").encode() not in parent
    before = show(MODEL)
    after = before.replace(
        "from sglang.srt.runtime_context import get_parallel, get_spec",
        "from sglang.srt.runtime_context import get_exec, get_parallel, get_spec",
    )
    marker = (
        "    def _transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:"
    )
    assert after.count(marker) == 1
    helper = (HERE / "compact_fc.py").read_text(encoding="utf-8-sig")
    helper_src = ast.get_source_segment(
        helper,
        next(
            n for n in ast.parse(helper).body if getattr(n, "name", "") == "compact_fc"
        ),
    )
    marker_class = "class DFlash2DraftModel(DFlashDraftModel):"
    after = after.replace(
        marker_class,
        helper_src.replace("def compact_fc(", "def _project_prefix_fc(")
        + "\n\n\n"
        + marker_class,
    )
    after = after.replace(
        marker,
        """    def project_accepted_target_hidden(self, hidden, lengths, width):
        if (
            self.quantizable
            or not isinstance(self.fc, nn.Linear)
            or hidden.dtype != torch.bfloat16
            or self.fc.weight.dtype != hidden.dtype
            or not hidden.is_cuda
            or torch.cuda.is_current_stream_capturing()
            or get_exec().deterministic.enable_deterministic_inference
            or hidden.ndim != 2
            or hidden.shape != (lengths.numel() * width, self.fc.in_features)
        ):
            return self.project_target_hidden(hidden)
        return self.hidden_norm(_project_prefix_fc(hidden, self.fc.weight, lengths, width))

"""
        + marker,
    )
    orig_worker = show(WORKER)
    line = (
        "            ctx_hidden = self.draft_model.project_target_hidden(target_hidden)"
    )
    assert orig_worker.count(line) == 1
    new_worker = orig_worker.replace(
        line,
        """            prefix_projection = getattr(
                self.draft_model, "project_accepted_target_hidden", None
            )
            if (
                prefix_projection is not None
                and cache_loc_2d is not None
                and commit_lens is not None
                and self.lilicorr is None
            ):
                ctx_hidden = prefix_projection(
                    target_hidden, commit_lens, int(cache_loc_2d.shape[1])
                )
            else:
                ctx_hidden = self.draft_model.project_target_hidden(target_hidden)""",
    )
    delta = patch(MODEL, before, after) + patch(WORKER, orig_worker, new_worker)
    file = HERE / "trial_parent_delta.diff"
    file.write_bytes(delta.encode())
    subprocess.run(
        ["git", "-C", str(REPO), "apply", "--check", "--whitespace=error", str(file)],
        check=True,
    )
    print("Trial patch applies cleanly on the two untouched pinned parent files.")
    print("Qualification required before enabling this trial.")


if __name__ == "__main__":
    main()
