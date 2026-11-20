"""Pack one exact collective in original DFlash2 global candidate selection."""
import ast,difflib,hashlib,os,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;D=H/"decode_selector";R=H.parent
pin="297e54fd105e974cdf401e5b4d20253a22c16f90"
base=(R/"release_2026-11-18/candidate.diff").read_bytes()
assert hashlib.sha256(base).hexdigest()=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
model="python/sglang/srt/models/dflash.py";mod="python/sglang/kernels/ops/speculative/dflash_packed_topk.py"
old=subprocess.check_output(["git","-C",str(R/"sglang"),"show",pin+":"+model]).decode()
new=old.replace("    global_ids = ids.long() + int(lm_head.shard_indices.org_vocab_start_index)",'''    padded_vocab = getattr(lm_head, "num_embeddings_padded", None)
    if (not with_partition and vals.is_cuda and not torch.version.hip
        and padded_vocab is not None and 0 < padded_vocab <= (1 << 32)):
        from sglang.kernels.ops.speculative.dflash_packed_topk import (
            pack_candidate_scores, pick_packed_candidates,
        )

        packed = pack_candidate_scores(vals, ids, int(lm_head.shard_indices.org_vocab_start_index))
        gathered = tensor_model_parallel_all_gather(packed, dim=-1)
        top_ids, top_vals = pick_packed_candidates(gathered, k)
        return top_ids, top_vals, None
    global_ids = ids.long() + int(lm_head.shard_indices.org_vocab_start_index)''')
assert new!=old
kernel=(H/"decode_packed_kernels.py").read_text(encoding="utf-8-sig")
for segment in ("    _pack_candidate_scores[","    _unpack_candidate_scores[","    _gather_selected_candidate_ids["):
    assert kernel.count(segment)==1
kernel=kernel.replace("    _pack_candidate_scores[","    if not ids.numel():return out\n    _pack_candidate_scores[")
kernel=kernel.replace("    _unpack_candidate_scores[","    if not packed.numel():\n        top,sel=torch.topk(values,k,dim=-1)\n        return sel,top\n    _unpack_candidate_scores[")
(D/"packed_kernels_checked.py").write_text(kernel,encoding="utf8")
ast.parse(new);ast.parse(kernel)
parts=[]
for file,a,b in [(mod,"",kernel),(model,old,new)]:
    assert ("diff --git a/"+file+" b/"+file).encode() not in base
    parts.append("diff --git a/"+file+" b/"+file+"\n"+("new file mode 100644\n" if not a else "")+
        "".join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),"a/"+file if a else "/dev/null","b/"+file)))
out=D/"packed_candidate.diff";out.write_bytes(base+"".join(parts).encode())
env={**os.environ,"GIT_INDEX_FILE":str(D/"packed.index")}
for args in (["read-tree",pin],["apply","--cached","--check",str(out)]):subprocess.run(["git","-C",str(R/"sglang"),*args],env=env,check=True)
print(len(out.read_bytes()),hashlib.sha256(out.read_bytes()).hexdigest())

