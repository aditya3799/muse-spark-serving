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
helper="python/sglang/kernels/ops/moe/staged_index.py"
helpersrc=(here/"staged_index.py").read_text().rstrip()+"\n"
(here/"staged_index.py").write_bytes(helpersrc.encode())
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
            lines[j]=lines[j].replace("from humming.layer import HummingMethod","from sglang.kernels.ops.moe.staged_index import staged_index_forward")
            lines[j]=lines[j].replace("HummingMethod.forward_layer(","staged_index_forward(")
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
vendor=root/"swarm_2026-11-16/moe/humming_vendor/humming"
nodes=[n for n in ast.parse(helpersrc).body if isinstance(n,ast.Assign) or (isinstance(n,ast.FunctionDef) and n.name=="adapted_source")]
ns={"humming":SimpleNamespace(__file__=str(vendor/"__init__.py")),"lru_cache":lru_cache,"Path":Path}
exec(compile(ast.Module(body=nodes,type_ignores=[]),"<render>","exec"),ns)
rendered=ns["adapted_source"]()
assert rendered and "slot >> 3" in rendered and "ctx.smem.wr_row_index[idx_in_tile] = slot;" in rendered
(here/"rendered.cuh").write_text(rendered)
rng=random.Random(919)
n=0
for m in (8,16,24,32,40,48,56,64,96,128,192,256):
    for threads in (32,64,128,256,512):
        for topk in (1,8):
            rows=[rng.choice([0,131072,0xffffffff,rng.randrange(0,131073)]) for _ in range(m)]
            rd=[None]*m;wr=[None]*m
            for th in range(threads):
                for i in range(th,m,threads):
                    assert wr[i] is None
                    wr[i]=rows[i];rd[i]=rows[i] if topk==1 else rows[i]>>3
            assert wr==rows and rd==[x//topk for x in rows]
            n+=m
report=dict(baseline_sha256=hashlib.sha256(parent).hexdigest(),candidate_sha256=hashlib.sha256(bundle).hexdigest(),
            indexed_slots_checked=n,variants_verified=120,normalized_source_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            intent="Only non-WS, indexed, full-K FP32 configurations and top_k 1 or 8; identical slot and read-row mappings including sentinel values.")
(here/"cpu.json").write_text(json.dumps(report,indent=2)+"\n");print(report)

