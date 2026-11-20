"""Build the decode target-vocabulary experiment over the exact scored parent, leaving the checkout untouched."""
import ast
import difflib
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
EXPECTED="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
LP="python/sglang/srt/layers/logits_processor.py"
MOD="python/sglang/kernels/ops/sampling/parallel_softcap_row_argmax.py"


def modify(src):
    imp="from sglang.kernels.ops.sampling.softcap_row_argmax import softcap_row_argmax"
    assert src.count(imp)==1
    src=src.replace(imp,imp+"\nfrom sglang.kernels.ops.sampling.parallel_softcap_row_argmax import parallel_softcap_max")
    old='''        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        if self._can_gather_softcap_fused(
            logits, local_hidden_states, lm_head, logits_metadata
        ):
            if (
                self.defer_verify_gather
                and logits_metadata.forward_mode.is_target_verify()
            ):
                return logits
            return self._gather_softcap_fused(
                logits, logits_metadata, use_logits_buffer
            )
'''
    new='''        can_fuse_gather = self._can_gather_softcap_fused(
            logits, local_hidden_states, lm_head, logits_metadata
        )
        deferred_verify = (
            can_fuse_gather
            and self.defer_verify_gather
            and logits_metadata.forward_mode.is_target_verify()
        )
        # The DFlash worker finishes a true TP shard after it knows whether
        # greedy selection or the complete distribution is required.
        if deferred_verify and logits.shape[-1] < self.vocab_size:
            return logits

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        if can_fuse_gather:
            if deferred_verify:
                return logits
            return self._gather_softcap_fused(
                logits, logits_metadata, use_logits_buffer
            )
'''
    assert src.count(old)==1
    src=src.replace(old,new)
    old='        return self._gather_softcap_fused(logits, None, use_buffer=False)'
    assert src.count(old)==1
    src=src.replace(old,'        if self.logit_scale is not None:\n            logits = logits * self.logit_scale\n'+old)
    old='''        local = softcap_row_argmax(
            logits, self.final_logit_softcapping, valid_width, col_offset
        )'''
    new='''        scale = 1.0 if self.logit_scale is None else self.logit_scale
        local = parallel_softcap_max(
            logits, self.final_logit_softcapping, valid_width, col_offset, scale=scale
        )'''
    assert src.count(old)==1
    src=src.replace(old,new).replace(imp+"\n","")
    return src


def main():
    patch=ROOT/"release_2026-11-18/candidate.diff"
    assert hashlib.sha256(patch.read_bytes()).hexdigest()==EXPECTED
    with tempfile.TemporaryDirectory() as tmp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/"index"))
        def git(*args):return subprocess.check_output(["git","-C",str(ROOT/"sglang"),*args],env=env)
        git("read-tree",BASE);git("apply","--cached","--whitespace=nowarn",str(patch))
        before=git("show",":"+LP).decode()
        after=modify(before)
        kern=(HERE/"sharded_max.py").read_text(encoding="utf-8").rstrip()+"\n"
        ast.parse(after);ast.parse(kern)
        lines=list(difflib.unified_diff(before.splitlines(True),after.splitlines(True),"a/"+LP,"b/"+LP))
        lines+=list(difflib.unified_diff([],kern.splitlines(True),"/dev/null","b/"+MOD))
        delta=HERE/"delta.diff";delta.write_bytes("".join(lines).encode())
        git("apply","--cached",str(delta))
        full=git("diff","--cached","--binary","--no-ext-diff",BASE);out=HERE/"candidate.diff";out.write_bytes(full)
        git("read-tree",BASE);git("apply","--cached","--check",str(out))
        assert len(full)<400000
        added=[l for l in full.splitlines() if l.startswith(b"+") and not l.startswith(b"+++")]
        assert max(len(l)-1 for l in added)<=400
        info=dict(sha256=hashlib.sha256(full).hexdigest(),baseline=EXPECTED,bytes=len(full),maximum_added_line_bytes=max(len(l)-1 for l in added),
                  paths=[LP,MOD],scope="Research candidate; timing, off-grid selection, output equivalence and target distributed performance must be checked before promotion.")
        (HERE/"package.json").write_text(json.dumps(info,indent=2));print(info)


if __name__=="__main__":main()