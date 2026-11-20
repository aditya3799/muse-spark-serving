"""Inspect contraction and operand order after the first two inverse square roots."""
from pathlib import Path
import json
import re
import hashlib
here=Path(__file__).resolve().parent
rows=json.loads((here/"offline.json").read_text())
cache=Path.home()/".triton/cache"
out=[]
for r in rows:
    if r["kernel"] not in ("_post_expert_norm_weighted_sum_kernel","_joined","_shard_final"):continue
    p=next(p for p in cache.glob(f"*/{r['kernel']}.cubin") if hashlib.sha256(p.read_bytes()).hexdigest()==r["cubin_sha256"])
    lines=p.with_suffix(".ptx").read_text().splitlines()
    for slot,i in enumerate([i for i,s in enumerate(lines) if "rsqrt.approx" in s]):
        after=lines[i+1:]
        # Collect arithmetic on any value downstream of this rsqrt before
        # the next rsqrt. Retain a representative lane for each form.
        roots={re.search(r"rsqrt.approx(?:.ftz)?.f32\s+(%r\d+)",lines[i]).group(1):"inv"}
        hits={}
        for s in after:
            if "rsqrt.approx" in s:break
            m=re.search(r"(mul\.f32|fma\.rn\.f32|add\.f32)\s+(%r\d+),\s*(%r\d+),\s*(%r\d+)(?:,\s*(%r\d+))?;",s)
            if not m:continue
            inputs=[m[3],m[4]]+([m[5]] if m[5] else [])
            if not any(reg in roots for reg in inputs):continue
            expr=m[1]+"("+",".join(roots.get(v,"external") for v in inputs)+")"
            roots[m[2]]=expr
            hits[expr]=hits.get(expr,0)+1
        out.append(dict(sm=r["sm"],kernel=r["kernel"],slot=slot,patterns=hits))
(here/"accum_tree.json").write_text(json.dumps(out,indent=2))
for r in out:
    if r["sm"]==86 and r["slot"] in (0,1):
        print(r["kernel"],r["slot"],r["patterns"],flush=True)
