"""Compare all per-slot FMA trees emitted by the pinned offline compiler."""
import hashlib
import json
import re
from pathlib import Path

here = Path(__file__).resolve().parent
common = (here / "warp_tree.py").read_text().split("tests=[]")[0]
common = common.replace("for line in prefix.splitlines():", """
    slot_trees=[]
    for line in text.splitlines():
        m=re.search(r"shfl.sync.bfly.b32\\s+%r\\d+,\\s*(%r\\d+),\\s*16,",line)
        if m:slot_trees.append(labels.get(m[1]))
""")
common = common.replace("return dict(thread_tree=labels[shuffles[0][0]]", "return dict(all_trees=slot_trees,thread_tree=labels[shuffles[0][0]]")
common=common.replace("assert len(vec)==4","assert len(vec)==(6 if stem=='_joined' else 4)")
exec(common)
rows.extend(json.loads((here/"tiny_offline.json").read_text()))

def orientations(tree):
    if tree[0]=="fma": return [(tree[1],tree[2][1])]
    if tree[0]=="add":return sorted(orientations(tree[1])+orientations(tree[2]))
    raise ValueError(tree)

tests=[]
for sm in [86,90]:
    for kernel in ["_post_expert_norm_weighted_sum_kernel","_post_expert_inv_rms_kernel","_square","_joined"]:
        row=next(r for r in rows if r["sm"]==sm and r["kernel"]==kernel)
        try:
            t=tree(row,kernel in ("_post_expert_norm_weighted_sum_kernel","_joined"))
            records=[dict(orientation=orientations(k),hash=hashlib.sha256(repr(k).encode()).hexdigest()) for k in t["all_trees"]]
            tests.append(dict(sm=sm,kernel=kernel,cubin=t["cubin"],slots=records))
            print(sm,kernel,len(records),[r["hash"][:6] for r in records],flush=True)
        except Exception as exc:
            tests.append(dict(sm=sm,kernel=kernel,error=repr(exc)))
            print(sm,kernel,repr(exc),flush=True)
(here/"all_slot_tree.json").write_text(json.dumps(tests,indent=2))
