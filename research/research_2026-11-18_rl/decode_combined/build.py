"""Assemble independently qualified decode changes against the frozen release by exact patched-source identity."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

here = Path(__file__).resolve().parent
root = here.parents[1]
pin = "297e54fd105e974cdf401e5b4d20253a22c16f90"
release = root/"release_2026-11-18/candidate.diff"
release_digest = "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
receipts = [
    (root/"research_2026-11-18_rl_prefill/decode_qkv/candidate.diff",root/"research_2026-11-18_rl_prefill/decode_qkv/entry_reward.json"),
    (root/"research_2026-11-18_window_repair/decode_selector/direct_candidate.diff",root/"research_2026-11-18_window_repair/decode_selector/selector_direct_walk_reward.json"),
]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    pages=parser.add_mutually_exclusive_group()
    pages.add_argument("--include-pages",action="store_true")
    pages.add_argument("--direct-pages",action="store_true")
    opts=parser.parse_args()
    prefix="direct_three_" if opts.direct_pages else ("three_" if opts.include_pages else "")
    if opts.direct_pages:
        base=root/"research_2026-11-18_rl_prefill/decode_pages/global_rows"
        receipts.append((base/"candidate.diff",base/"direct_reward.json"))
    if opts.include_pages:
        base=root/"research_2026-11-18_rl_prefill/decode_pages"
        receipts.append((base/"candidate.diff",base/"metadata_reward.json"))
    assert digest(release)==release_digest
    result={"base":pin,"release_sha256":release_digest,"components":[]}
    with tempfile.TemporaryDirectory() as tmp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/"index"))
        def git(*args,raw=None):
            return subprocess.run(["git","-C",str(root/"sglang"),*args],env=env,input=raw,capture_output=True,check=True).stdout
        git("read-tree",pin)
        git("apply","--cached","--whitespace=nowarn",str(release))
        parent=git("write-tree").strip().decode()
        occupied=set()
        pieces=[]
        for patch,receipt in receipts:
            score=json.loads(receipt.read_text(encoding="utf-8-sig"))
            assert score["local_keep"] and score["baseline_sha256"]==release_digest
            assert all(run["decision"]=="KEEP" and run["controlled"] for run in score["runs"])
            git("read-tree",pin)
            git("apply","--cached","--whitespace=nowarn",str(patch))
            paths=git("diff","--cached","--name-only",parent).decode().splitlines()
            assert paths and all(r["candidate_sha256"]==digest(patch) for r in score["runs"])
            assert occupied.intersection(paths)<={"python/sglang/srt/layers/attention/flashattention_backend.py"}
            occupied.update(paths)
            delta=git("diff","--cached","--binary",parent)
            pieces.append(delta)
            result["components"].append(dict(patch=patch.relative_to(root).as_posix(),sha256=digest(patch),receipt=receipt.relative_to(root).as_posix(),
                                             measured_sha256=[r["candidate_sha256"] for r in score["runs"]],paths=paths))
        git("read-tree",parent)
        for delta in pieces:
            git("apply","--cached","--whitespace=nowarn","-",raw=delta)
        merged=git("write-tree")
        git("read-tree",parent)
        for delta in reversed(pieces):
            git("apply","--cached","--whitespace=nowarn","-",raw=delta)
        assert git("write-tree")==merged, "component edits depend on apply order"
        for path in occupied:
            assert path.startswith("python/sglang/") and path.endswith(".py")
            ast.parse(git("show",":"+path).decode("utf-8"))
        complete=git("diff","--cached","--binary",pin)
        assert len(complete)<=400000
        max_line=max(len(line)-1 for line in complete.splitlines() if line.startswith(b"+") and not line.startswith(b"+++"))
        assert max_line<=400
        outfile=here/(prefix+"candidate.diff");outfile.write_bytes(complete)
        git("read-tree",pin)
        git("apply","--cached","--check","--whitespace=nowarn",str(outfile))
        result.update(sha256=digest(outfile),bytes=len(complete),max_added_line=max_line,scope="Clean-apply composition of independently timed decode stages. Joint server speed and target Hopper activation require hardware verification.")
    (here/(prefix+"manifest.json")).write_text(json.dumps(result,indent=2))
    print(result["sha256"],result["bytes"],list(sorted(occupied)))

if __name__=="__main__":main()