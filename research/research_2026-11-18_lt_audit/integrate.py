"""Construct and apply-check the cumulative source-only startup-warmed DFlash projection patch."""
import ast
import difflib
import hashlib
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=ROOT/"sglang"
BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
DIGEST="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
parent=(ROOT/"release_2026-11-18/candidate.diff").read_bytes()
assert hashlib.sha256(parent).hexdigest()==DIGEST
out=HERE/"integrated"
out.mkdir(exist_ok=True)
paths={}
def old(path):
    return subprocess.check_output(["git","-C",str(REPO),"show",BASE+":"+path]).decode()
def one(text,a,b):
    assert text.count(a)==1,a
    return text.replace(a,b)
def add(path,before,after):
    if path.endswith(".py"): ast.parse(after)
    if before is None:
        diff="".join(difflib.unified_diff([],after.splitlines(True),"/dev/null","b/"+path))
        diff="diff --git a/"+path+" b/"+path+"\nnew file mode 100644\n"+diff
    else:
        assert ("diff --git a/"+path+" b/"+path).encode() not in parent
        diff="diff --git a/"+path+" b/"+path+"\n"+"".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),"a/"+path,"b/"+path))
    target=out/path
    target.parent.mkdir(exist_ok=True,parents=True)
    target.write_bytes(after.encode())
    paths[path]=diff
def read(path): return path.read_text(encoding="utf-8-sig")
package="python/sglang/srt/speculative/full_projector/"
for file in ("__init__.py","backend.py","vendor.cc"):
    contents=read(HERE/"full_projector"/file)
    if file=="backend.py":
        contents=one(contents,"from pathlib import Path","from pathlib import Path\nfrom .build_helper import native_binary")
        contents=one(contents,'ct.CDLL(str(HERE / "sweep.so"))','ct.CDLL(str(native_binary()))')
    add(package+file,None,contents)
add(package+"build_helper.py",None,read(HERE/"production_loader.py"))
add(package+"warmup.py",None,read(HERE/"production_warmup.py"))
model="python/sglang/srt/models/dflash.py"
before=old(model)
after=one(before,"        projected = self.fc(target_hidden)\n",
    "        projector = getattr(self, \"_context_projector\", None) if target_hidden.shape[0] >= 128 else None\n"
    "        projected = self.fc(target_hidden) if projector is None else projector(target_hidden)\n")
after=one(after,'    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):\n','    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):\n        projector = getattr(self, "_context_projector", None)\n        if projector is not None:\n            projector.close()\n            self._context_projector = None\n')
add(model,before,after)
worker="python/sglang/srt/speculative/dflash_worker_v2.py"
before=old(worker)
marker="        if not is_cuda() or not is_dflash_sampling_verify_available():\n            return self.model_runner.prewarm_sampling()\n"
after=one(before,marker,'''        if is_cuda() and not get_exec().deterministic.enable_deterministic_inference:
            from sglang.srt.speculative.full_projector.warmup import prewarm_projector

            with draft_tp_context(self.draft_parallel_layout):
                prewarm_projector(
                    self.draft_model,
                    get_exec().graph.cuda_graph_config.decode.bs,
                    self.block_size,
                    get_parallel().tp_group,
                )
'''+marker)
add(worker,before,after)
patch=HERE/"integrated_candidate.diff"
increment="".join(paths.values())
assert all(len(line)<=400 for line in increment.splitlines() if line.startswith("+"))
payload=parent+increment.encode()
all_lines=payload.decode("utf-8").splitlines()
added=[line[1:] for line in all_lines if line.startswith("+") and not line.startswith("+++")]
assert max(map(len,added))<=400
for line in all_lines:
    if line.startswith("diff --git "):
        left,right=line.split()[2:]
        assert left[2:]==right[2:] and left.startswith("a/python/sglang/")
        assert "/srt/model_loader/" not in left
assert not any(line.startswith(("rename from ","rename to ","GIT binary patch")) for line in all_lines)
print("full-patch max added line",max(map(len,added)))
assert len(payload)<=400000
patch.write_bytes(payload)
index=HERE/"integration_check.index"
env={**os.environ,"GIT_INDEX_FILE":str(index)}
subprocess.run(["git","-C",str(REPO),"read-tree",BASE],check=True,env=env)
subprocess.run(["git","-C",str(REPO),"apply","--cached","--check",str(patch)],check=True,env=env)
print("cumulative source patch",len(payload),hashlib.sha256(payload).hexdigest())

