"""Patch captured selector so the original walk directly writes its public output buffers."""
import ast,hashlib,os,subprocess,difflib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent
D=H/"decode_selector";D.mkdir(exist_ok=True)
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90";SHA="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
parent=(R/"release_2026-11-18/candidate.diff").read_bytes();assert hashlib.sha256(parent).hexdigest()==SHA
parts=[]
def patch(file,reps):
    old=subprocess.check_output(["git","-C",str(R/"sglang"),"show",PIN+":"+file]).decode()
    assert ("diff --git a/"+file+" b/"+file).encode() not in parent
    new=old
    for a,b in reps:
        assert new.count(a)==1,(file,a)
        new=new.replace(a,b)
    ast.parse(new)
    parts.append("diff --git a/"+file+" b/"+file+"\n"+"".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),"a/"+file,"b/"+file)))
    (D/(Path(file).stem+"_original_src.py")).write_text(old,encoding="utf8")
    (D/(Path(file).stem+"_direct_src.py")).write_text(new,encoding="utf8")
    return old,new
kernel="python/sglang/kernels/ops/speculative/dflash.py"
old,new=patch(kernel,[("    q_ptr,\n    slots: tl.constexpr,\n    top_k: tl.constexpr,\n):","    q_ptr,\n    slots: tl.constexpr,\n    top_k: tl.constexpr,\n    candidate_out_ptr=None,\n):"),
    ("        tl.store(q_ptr + base + offsets, probabilities)","        if candidate_out_ptr is not None:\n            tl.store(candidate_out_ptr + base + offsets, tl.load(candidate_ptr + base + offsets))\n        tl.store(q_ptr + base + offsets, probabilities)"),
    ("    greedy_mask,\n):\n    batch, slots, top_k = candidate_ids.shape","    greedy_mask,\n    tokens=None,\n    q_rows=None,\n    candidate_out=None,\n):\n    batch, slots, top_k = candidate_ids.shape"),
    ("    tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)\n    q_rows = torch.empty(","    if tokens is None:\n        tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)\n    if q_rows is None:\n        q_rows = torch.empty("),
    ("        (batch, slots, top_k), dtype=torch.float32, device=scores.device\n    )\n    _selector_walk_kernel","            (batch, slots, top_k), dtype=torch.float32, device=scores.device\n        )\n    _selector_walk_kernel"),
    ("        top_k=top_k,\n        num_warps=1,\n    )\n    return tokens, q_rows","        top_k=top_k,\n        candidate_out_ptr=candidate_out,\n        num_warps=1,\n    )\n    return tokens, q_rows")])
for name,source in (("original",old),("direct",new)):
    nodes={n.name:n for n in ast.parse(source).body if hasattr(n,"name")}
    out=["import torch\nimport triton\nimport triton.language as tl\n"]
    lines=source.splitlines()
    for key in ("_selector_walk_kernel","selector_walk_triton"):
        n=nodes[key];first=n.decorator_list[0].lineno if n.decorator_list else n.lineno
        out.append("\n".join(lines[first-1:n.end_lineno]))
    (D/(name+"_walk.py")).write_text("\n\n".join(out)+"\n",encoding="utf8")
model="python/sglang/srt/models/dflash.py"
anchor="    def sample_path(\n"
patch(model,[(anchor,'''    def sample_path_into(self, *, candidate_ids, scores, uniforms, temperatures,
                         greedy_mask, tokens, q_rows, candidate_out):
        return selector_walk_triton(
            candidate_ids=candidate_ids, scores=scores, uniforms=uniforms,
            temperatures=temperatures, greedy_mask=greedy_mask, tokens=tokens,
            q_rows=q_rows, candidate_out=candidate_out)

'''+anchor)])
worker="python/sglang/srt/speculative/dflash_worker_v2.py"
anchor="        # In-graph philox draw: each replay advances the generator and redraws.\n"
patch(worker,[(anchor,anchor+'''        if scores.is_cuda and hasattr(self.selector, "sample_path_into"):
            self.selector.sample_path_into(
                candidate_ids=candidate_ids, scores=scores,
                uniforms=self.uniforms[:bs].uniform_(),
                temperatures=self.temperatures[:bs], greedy_mask=self.greedy_mask[:bs],
                tokens=self.out[:bs * (self.block_size - 1)].view(bs, self.block_size - 1),
                q_rows=self.q_out[:bs], candidate_out=self.candidate_out[:bs],
            )
            return
''')])
out=D/"direct_candidate.diff";out.write_bytes(parent+"".join(parts).encode())
env={**os.environ,"GIT_INDEX_FILE":str(D/"direct.index")}
for args in (["read-tree",PIN],["apply","--cached","--check",str(out)]):subprocess.run(["git","-C",str(R/"sglang"),*args],env=env,check=True)
assert max(len(x[1:]) for x in out.read_bytes().splitlines() if x.startswith(b"+"))<=400
print(len(out.read_bytes()),hashlib.sha256(out.read_bytes()).hexdigest())

