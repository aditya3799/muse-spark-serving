import ast
import difflib
import hashlib
import json
import random
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

here=Path(__file__).resolve().parent
root=here.parents[1]
parent=(root/"release_2026-11-18/candidate.diff").read_bytes()
base="297e54fd105e974cdf401e5b4d20253a22c16f90"
assert hashlib.sha256(parent).hexdigest()=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
helper="python/sglang/kernels/ops/moe/scale_order.py"
helpersrc=(here/"scale_order.py").read_text().rstrip()+"\n"
(here/"scale_order.py").write_bytes(helpersrc.encode())
def diff(name,old,new):
    return "diff --git a/{0} b/{0}\n".format(name)+("" if old else "new file mode 100644\n")+ "".join(
        difflib.unified_diff(old.splitlines(True),new.splitlines(True),"a/"+name if old else "/dev/null","b/"+name))
sections=re.split(r"(?=^diff --git )",parent.decode(),flags=re.MULTILINE)[1:]
delta=[]
originals={}
for i,section in enumerate(sections):
    name=section.splitlines()[0].split()[2][2:]
    run=subprocess.run(["git","-C",str(root/"sglang"),"show",f"{base}:{name}"],capture_output=True)
    originals[name]=run.stdout.decode() if not run.returncode else ""
    if not (name.endswith("muse_spark_humming_split.py") or name.endswith("muse_spark_nvfp4_moe.py")):
        continue
    scored=(root/"release_2026-11-18/candidate_sources"/name).read_text()
    lines=scored.splitlines(True)
    funcs=0
    calls=0
    for node in ast.parse(scored).body:
        if not isinstance(node,ast.FunctionDef):
            continue
        segment=lines[node.lineno-1:node.end_lineno]
        count=sum(s.count("HummingMethod.forward_layer(") for s in segment)
        if not count:
            continue
        funcs+=1;calls+=count
        for j in range(node.lineno-1,node.end_lineno):
            lines[j]=lines[j].replace("from humming.layer import HummingMethod","from sglang.kernels.ops.moe.scale_order import scale_order_forward")
            lines[j]=lines[j].replace("HummingMethod.forward_layer(","scale_order_forward(")
    assert (funcs,calls)==((2,4) if name.endswith("split.py") else (1,2))
    changed="".join(lines)
    ast.parse(changed)
    (here/Path(name).name).write_bytes(changed.encode())
    delta.append(diff(name,scored,changed))
    sections[i]=diff(name,originals[name],changed)
sections.append(diff(helper,"",helpersrc))
delta.append(sections[-1])
with tempfile.TemporaryDirectory(dir=here) as t:
    tree=Path(t)
    subprocess.run(["git","-C",t,"init","-q"],check=True)
    def reset():
        for name,old in originals.items():
            p=tree/name
            if old:
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(old.encode())
            elif p.exists():p.unlink()
    def apply(data,*flags):
        subprocess.run(["git","-C",t,"apply","--whitespace=error",*flags,"-"],input=data,check=True)
    reset();apply(parent)
    for name in originals:
        assert (tree/name).read_bytes()==(root/"release_2026-11-18/candidate_sources"/name).read_bytes()
    apply("".join(delta).encode(),"--check");reset()
    bundle="".join(sections).encode()
    apply(bundle,"--check");apply(bundle)
    (here/"candidate.diff").write_bytes(bundle)
    (here/"delta.diff").write_bytes("".join(delta).encode())
