"""Assemble and replay a complete patch against the pinned scored parent."""
import ast
import difflib
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
release = root / "release_2026-11-18"
parent = (release / "candidate.diff").read_bytes()
assert hashlib.sha256(parent).hexdigest() == "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
base = "297e54fd105e974cdf401e5b4d20253a22c16f90"
sections = re.split(r"(?=^diff --git )", parent.decode(), flags=re.MULTILINE)[1:]
wrapper = "python/sglang/kernels/ops/moe/route_activation.py"
header = "python/sglang/kernels/jit/csrc/elementwise/route_activation.cuh"

def difference(file, before, after):
    old = "a/"+file if before else "/dev/null"
    return "diff --git a/{0} b/{0}\n".format(file)+("" if before else "new file mode 100644\n")+ "".join(
        difflib.unified_diff(before.splitlines(True),after.splitlines(True),old,"b/"+file))

changes = []
originals = {}
for i, section in enumerate(sections):
    name = section.splitlines()[0].split()[2][2:]
    run = subprocess.run(["git","-C",str(root/"sglang"),"show",f"{base}:{name}"],capture_output=True)
    old = run.stdout.decode() if not run.returncode else ""
    originals[name] = old
    if not (name.endswith("muse_spark_nvfp4_moe.py") or name.endswith("muse_spark_humming_split.py")):
        continue
    scored = (release/"candidate_sources"/name).read_bytes().decode()
    # Sources and patched base must match byte-for-byte; verified below.
    updated = scored.replace("    from sglang.kernels.ops.activation.activation import silu_and_mul\n",
        "    from sglang.kernels.ops.moe.route_activation import scaled_prefill_activation\n")
    updated, count = re.subn(
        r"    scale_active_rows\(cache1,\s*local_ids,\s*layer.w13_post_scale\)\n"
        r"(    cache2 = torch.empty\([^\n]+\))\n"
        r"    silu_and_mul\(cache1,\s*cache2,\s*expert_ids=local_ids.flatten\(\),\s*expert_step=1\)",
        r"\1\n    scaled_prefill_activation(cache1, cache2, local_ids, layer.w13_post_scale)", updated)
    assert count == (2 if name.endswith("split.py") else 1), (name,count)
    ast.parse(updated)
    changes.append(difference(name,scored,updated))
    sections[i] = difference(name,old,updated)
for name in (header,wrapper):
    path = here / Path(name).name
    source = path.read_text().rstrip() + "\n"
    path.write_bytes(source.encode())
    if name.endswith(".py"):
        ast.parse(source)
    new = difference(name,"",source)
    changes.append(new)
    sections.append(new)

with tempfile.TemporaryDirectory(dir=here) as temporary:
    tree = Path(temporary)
    subprocess.run(["git","-C",str(tree),"init","-q"],check=True)
    def reset():
        for name, original in originals.items():
            path=tree/name
            if original:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(original.encode())
            elif path.exists():
                path.unlink()
    def apply(text, *flags):
        subprocess.run(["git","-C",str(tree),"apply","--whitespace=error",*flags,"-"],input=text,check=True)
    reset()
    apply(parent)
    for name in originals:
        assert (tree/name).read_bytes()==(release/"candidate_sources"/name).read_bytes(), name
    delta="".join(changes).encode()
    apply(delta,"--check")
    reset()
    full="".join(sections).encode()
    apply(full,"--check")
    apply(full)
    (here/"candidate.diff").write_bytes(full)
    (here/"delta.diff").write_bytes(delta)
    print("Packaged",len(full),hashlib.sha256(full).hexdigest())


