"""Build a clean incremental and standalone patch from the exact scored release."""
import ast
import difflib
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / "release_2026-11-18"
MODEL = "python/sglang/srt/models/muse_spark.py"
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
DIGEST = "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"


def change(text):
    replacements = [
        ("        aux_hidden_states = []\n        for layer_idx",
         """        aux_hidden_states = []
        packed_aux = None
        write_packed = (
            len(self.layers_to_capture) > 1
            and self.residual_dtype != self.operator_dtype
            and len(set(self.layers_to_capture)) == len(self.layers_to_capture)
        )
        tap_offset = 0
        for layer_idx"""),
        ("                aux_hidden_states.append(stream.to(self.operator_dtype))",
         """                if write_packed:
                    width = stream.shape[-1]
                    if packed_aux is None:
                        packed_aux = stream.new_empty(
                            (*stream.shape[:-1], width * len(self.layers_to_capture)),
                            dtype=self.operator_dtype,
                        )
                    packed_aux[..., tap_offset : tap_offset + width].copy_(stream)
                    tap_offset += width
                else:
                    aux_hidden_states.append(stream.to(self.operator_dtype))"""),
        ("        if aux_hidden_states:\n            return hidden_states, aux_hidden_states",
         "        if packed_aux is not None:\n            return hidden_states, packed_aux\n        if aux_hidden_states:\n            return hidden_states, aux_hidden_states"),
    ]
    for before, after in replacements:
        assert text.count(before) == 1
        text = text.replace(before, after)
    ast.parse(text)
    return text


def main():
    patch = PARENT / "candidate.diff"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == DIGEST
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        def git(*args):
            return subprocess.check_output(["git", "-C", str(ROOT / "sglang"), *args], env=env)
        git("read-tree", BASE)
        git("apply", "--cached", "--whitespace=nowarn", str(patch))
        parent = git("show", ":" + MODEL).decode()
        assert parent == (PARENT / "candidate_sources" / MODEL).read_text(encoding="utf-8")
        modified = change(parent)
        dest = HERE / "aux_increment.diff"
        dest.write_bytes("".join(difflib.unified_diff(parent.splitlines(True), modified.splitlines(True),
                         "a/"+MODEL, "b/"+MODEL, n=3)).encode())
        git("apply", "--cached", str(dest))
        bundled = git("diff", "--cached", "--binary", "--no-ext-diff", BASE)
        (HERE / "aux_candidate.diff").write_bytes(bundled)
        git("read-tree", BASE)
        git("apply", "--check", "--cached", str(HERE / "aux_candidate.diff"))
        print("Patch applies. SHA256", hashlib.sha256(bundled).hexdigest())


if __name__ == "__main__":
    main()
