"""Rebase two targeted edits onto the exact scored tree and validate the combined patch."""
import ast
import difflib
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
repo = root / "sglang"
base = "297e54fd105e974cdf401e5b4d20253a22c16f90"
parent = (root / "release_2026-11-18/candidate.diff").read_bytes()
assert hashlib.sha256(parent).hexdigest() == "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
helper = "python/sglang/srt/layers/communication/muse_spark_gemm_ag.py"
model = "python/sglang/srt/models/muse_spark.py"

def update(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

def diff(before, after, file):
    return f"diff --git a/{file} b/{file}\n" + "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), "a/" + file, "b/" + file))

def edit(file, source):
    if file == helper:
        return update(source, """    local_fp32 = torch.mm(
        x[start : start + m_local], weight.t(), out_dtype=torch.float32
    )
    local = local_fp32.to(x.dtype)
""", """    return _project_and_gather_rows(group, x[start : start + m_local], weight)


def reduce_scatter_prefill_row_projection(group, partials, weight):
    \"\"\"Reduce contiguous rank-owned rows before applying the replicated weight.\"\"\"
    local = partials.new_empty((partials.shape[0] // group.world_size, partials.shape[1]))
    group.reduce_scatter_tensor(local, partials)
    return _project_and_gather_rows(group, local, weight)


def _project_and_gather_rows(group, rows, weight):
    local_fp32 = torch.mm(rows, weight.t(), out_dtype=torch.float32)
    local = local_fp32.to(rows.dtype)
""")
    source = update(source, "    prefill_row_projection_supported,\n",
                    "    prefill_row_projection_supported,\n    reduce_scatter_prefill_row_projection,\n")
    return update(source, """            and not already_reduced
        ):
            output = tensor_model_parallel_all_reduce(output)
        if self.post_expert_proj is not None:""", """            and not already_reduced
        ):
            if self.post_expert_proj is not None and output.is_contiguous():
                weight = self.post_expert_proj.weight
                if self._can_shard_prefill_row_projection(output, weight):
                    group = get_parallel().tp_group
                    if not gemm_ag_supported(group.ca_comm, output, weight):
                        return reduce_scatter_prefill_row_projection(group, output, weight)
            output = tensor_model_parallel_all_reduce(output)
        if self.post_expert_proj is not None:""")

sections = re.split(r"(?=^diff --git )", parent.decode(), flags=re.MULTILINE)[1:]
with tempfile.TemporaryDirectory(dir=here) as temp:
    tree = Path(temp)
    originals = {}
    def reset_base():
        for entry in sections:
            file = entry.splitlines()[0].split()[2][2:]
            path = tree / file
            result = subprocess.run(["git", "-C", str(repo), "show", f"{base}:{file}"],
                                    capture_output=True)
            if result.returncode == 0:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(result.stdout)
                originals[file] = result.stdout.decode()
            elif path.exists():
                path.unlink()
    def apply(*flags, patch):
        subprocess.run(["git", "-C", str(tree), "apply", "--whitespace=error", *flags, "-"],
                       input=patch, check=True)
    subprocess.run(["git", "-C", str(tree), "init", "-q"], check=True)
    reset_base()
    apply(patch=parent)
    delta = []
    for i, entry in enumerate(sections):
        file = entry.splitlines()[0].split()[2][2:]
        if file in (helper, model):
            source = (tree / file).read_bytes().decode()
            modified = edit(file, source)
            ast.parse(modified)
            delta.append(diff(source, modified, file))
            sections[i] = diff(originals[file], modified, file)
            (here / Path(file).name).write_bytes(modified.encode())
    changes = "".join(delta).encode()
    apply("--check", patch=changes)
    reset_base()
    bundle = "".join(sections).encode()
    apply("--check", patch=bundle)
    apply(patch=bundle)
    for file in (helper, model):
        assert (here / Path(file).name).read_bytes() == (tree / file).read_bytes()
    (here / "candidate.diff").write_bytes(bundle)
    (here / "delta.diff").write_bytes(changes)
    print("Combined candidate", hashlib.sha256(bundle).hexdigest(), len(bundle))


