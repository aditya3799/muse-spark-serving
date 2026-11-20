"""Package the selected graph-owned snapshot arm on the exact scored release."""
import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

here=Path(__file__).resolve().parent
root=here.parents[1]
earlier=root/"research_2026-11-18_rl_prefill/decode_pages"
repo=root/"sglang"
pin="297e54fd105e974cdf401e5b4d20253a22c16f90"
meta="python/sglang/kernels/ops/attention/metadata.py"
backend="python/sglang/srt/layers/attention/flashattention_backend.py"
kernel="python/sglang/kernels/ops/attention/muse_fa3_page_snapshot.py"
def unified(path,before,after,new=False):
    hdr=f"diff --git a/{path} b/{path}\n"+("new file mode 100644\n" if new else "")
    return hdr+"".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),"/dev/null" if new else "a/"+path,"b/"+path))
def parts(body):
    return {c.splitlines()[0].split()[2][2:]:c for c in re.split(r"(?=^diff --git )",body,flags=re.M) if c}
def run(*cmd,**args):return subprocess.run(cmd,check=True,capture_output=True,**args)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--arm",help="use discovery winner by default");args=parser.parse_args()
    arm=args.arm or json.loads((here/"selection.json").read_text())["selected"]
    match=re.fullmatch(r"b(1024|2048|4096|8192)w(4|8)",arm);assert match
    block,warps=map(int,match.groups())
    parent=(root/"release_2026-11-18/candidate.diff").read_bytes()
    assert hashlib.sha256(parent).hexdigest()=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    original=parts(parent.decode());changes=original.copy()
    body=(earlier/"metadata.py").read_text(encoding="utf-8")
    old="    batch_size = cache_seqlens_int32.shape[0]\n    # Lengths-only"
    dispatch="""    if (
        swa_suffix_tokens and page_size == 1 and max_seq_pages > 0
        and full_v2p is None and swa_v2p is None
        and cache_seqlens_int32.numel() > 0 and req_to_token.dtype == torch.int32
        and all(t.is_contiguous() for t in (
            cache_seqlens_int32, cu_seqlens_k, seq_lens, req_pool_indices,
            full_to_swa_index_mapping,
        ))
        and all(t.stride(1) == 1 for t in (page_table, req_to_token, swa_page_table))
    ):
        from sglang.kernels.ops.attention.muse_fa3_page_snapshot import snapshot
        return snapshot(
            cache_seqlens_int32, cu_seqlens_k, page_table, req_to_token, req_pool_indices,
            max_seq_pages, seq_lens, seq_len_delta, page_size, swa_page_table,
            full_to_swa_index_mapping, swa_suffix_tokens=swa_suffix_tokens,
            full_block=BLOCK_CHOSEN, warps=WARPS_CHOSEN,
        )
""".replace("BLOCK_CHOSEN",str(block)).replace("WARPS_CHOSEN",str(warps))
    assert body.count(old)==1
    updated={meta:body.replace(old,dispatch+old),backend:(earlier/"flashattention_backend.py").read_text(),kernel:(here/"snapshot.py").read_text()}
    delta=[]
    with tempfile.TemporaryDirectory(dir=here) as temp:
        directory=Path(temp);run("git","init","-q",str(directory))
        for path,after in updated.items():
            ast.parse(after)
            is_new=path==kernel
            pristine="" if is_new else run("git","-C",str(repo),"show",pin+":"+path).stdout.decode()
            dest=directory/path
            if not is_new:
                dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(pristine.encode())
                if path in original:run("git","-C",temp,"apply","-",input=original[path].encode())
            before="" if is_new else dest.read_bytes().decode()
            delta.append(unified(path,before,after,is_new))
            changes[path]=unified(path,pristine,after,is_new)
        inc="".join(delta).encode()
        run("git","-C",temp,"apply","--check","-",input=inc)
        run("git","-C",temp,"apply","-",input=inc)
        assert all((directory/p).read_bytes()==v.encode() for p,v in updated.items())
    cumulative="".join(changes.values()).encode()
    with tempfile.TemporaryDirectory(dir=here) as t:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(t)/"validation.index"))
        run("git","-C",str(repo),"read-tree",pin,env=env)
        run("git","-C",str(repo),"apply","--cached","--check","-",input=cumulative,env=env)
    assert len(cumulative)<=400000
    for p,source in updated.items():
        if p==kernel:assert max(map(len,source.splitlines()))<=400
    (here/"delta.diff").write_bytes(inc);(here/"candidate.diff").write_bytes(cumulative)
    manifest=dict(arm=arm,source_pin=pin,old_parent_sha256=hashlib.sha256(parent).hexdigest(),
        candidate_sha256=hashlib.sha256(cumulative).hexdigest(),cumulative_bytes=len(cumulative),
        delta_sha256=hashlib.sha256(inc).hexdigest(),original_early_scheduler_release=True,changes=list(updated))
    (here/"manifest.json").write_text(json.dumps(manifest,indent=2));print(manifest)
if __name__=="__main__":main()
