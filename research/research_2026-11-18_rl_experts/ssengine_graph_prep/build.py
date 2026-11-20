"""Build both DFlash edits against the scored release in a temporary index."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

here = Path(__file__).resolve().parent
root = here.parents[1]
repo = root / "sglang"
base = "297e54fd105e974cdf401e5b4d20253a22c16f90"
worker = "python/sglang/srt/speculative/dflash_worker_v2.py"
info = "python/sglang/srt/speculative/dflash_info.py"
release = (root / "release_2026-11-18/candidate.diff").read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


assert sha(release) == "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
with tempfile.TemporaryDirectory() as tmp:
    env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}

    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", str(repo), *args], input=data, env=env)

    git("read-tree", base)
    git("apply", "--cached", "--whitespace=nowarn", "-", data=release)
    parent = git("write-tree").decode().strip()
    for path in (info, worker):
        code = git("show", parent + ":" + path).decode()
        if path == info:
            code = replace_once(code,
                "        defer_metadata_init: bool = False,\n",
                "        defer_metadata_init: bool = False,\n        defer_graph_init: bool = False,\n")
            code = replace_once(code, "        elif can_run_cuda_graph:\n",
                "        elif can_run_cuda_graph and defer_graph_init:\n"
                "            # The target runner stages graph inputs and metadata at replay.\n"
                "            return verify_forward_batch, can_run_cuda_graph\n"
                "        elif can_run_cuda_graph:\n")
        else:
            code = replace_once(code, "        verify_forward_batch, _ = verify_input.prepare_for_verify(\n",
                "        verify_forward_batch, graph_eligible = verify_input.prepare_for_verify(\n")
            code = replace_once(code, "            defer_metadata_init=get_parallel().attn_dp_enabled,\n",
                "            defer_metadata_init=get_parallel().attn_dp_enabled,\n"
                "            defer_graph_init=is_cuda(),\n")
            code = replace_once(code, "                None if _is_npu or get_parallel().attn_dp_enabled else True\n",
                "                None\n"
                "                if _is_npu or get_parallel().attn_dp_enabled or (is_cuda() and graph_eligible)\n"
                "                else True\n")
        compile(code, path, "exec")
        obj = git("hash-object", "-w", "--stdin", data=code.encode()).decode().strip()
        git("update-index", "--cacheinfo", "100644", obj, path)
    target = git("write-tree").decode().strip()
    delta = git("diff", "--binary", parent, target)
    cumulative = git("diff", "--binary", base, target)
    git("read-tree", base)
    git("apply", "--cached", "--whitespace=nowarn", "-", data=cumulative)
    assert git("write-tree").decode().strip() == target
    (here / "incremental.diff").write_bytes(delta)
    (here / "candidate.diff").write_bytes(cumulative)
    manifest = dict(base=base, parent_sha256=sha(release), increment_sha256=sha(delta),
        candidate_sha256=sha(cumulative), parent_tree=parent, target_tree=target, source_paths=[info,worker],
        upstream="https://github.com/SyFI-ServingStudio/SSEngine-GLM5.3-Flash/blob/e17de72b4e918da1adfb60854f0bcd04828821b7/ssengine/model/stage.py#L676",
        attribution="Compared static-input staging before replay to SSEngine Stage._run_graph. No upstream implementation copied.")
    (here / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(sha(cumulative), len(delta))
