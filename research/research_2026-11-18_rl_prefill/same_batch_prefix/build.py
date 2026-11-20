from pathlib import Path
import difflib
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
POLICY = "python/sglang/srt/managers/schedule_policy.py"
SCHED = "python/sglang/srt/managers/scheduler.py"

METHOD = '''    def shares_admitted_prefix(self, req: Req) -> bool:
        """Wait for an admitted request to publish a substantial part of this extension."""
        if (
            not self.can_run_list
            or req.host_hit_length
            or req.skip_radix_cache_insert
            or req.is_dllm()
            or req.return_logprob
            or self.tree_cache.supports_mamba()
            or not self.tree_cache.supports_prefix_sharing()
        ):
            return False

        ids = req.full_untruncated_fill_ids
        start = req.prefix_len
        budget = self.rem_chunk_tokens
        pending = len(ids) - start
        if budget is not None:
            pending = min(pending, budget)
        # Delay only when at least half of the work that fits could be reused.
        tokens = max(IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD, (pending + 1) // 2)
        end = ((start + max(tokens, 1) + self.page_size - 1) // self.page_size) * self.page_size
        if end >= len(ids) or end > start + pending:
            return False

        for earlier in self.can_run_list:
            if (
                earlier.skip_radix_cache_insert
                or earlier.host_hit_length
                or earlier.prefix_len != start
                or earlier.extend_end < end
                or earlier.last_node != req.last_node
                or earlier.extra_key != req.extra_key
                or earlier.cache_salt != req.cache_salt
            ):
                continue
            other = earlier.full_untruncated_fill_ids
            if ids[start] == other[start] and ids[end - 1] == other[end - 1]:
                if ids[start:end] == other[start:end]:
                    return True
        return False

'''

def base(path):
    return subprocess.check_output(["git", "-C", str(ROOT / "sglang"), "show", BASE + ":" + path]).decode()

def main():
    scored = (ROOT / "release_2026-11-18" / "candidate.diff").read_bytes()
    assert hashlib.sha256(scored).hexdigest() == "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    pieces = []
    for path, marker, insert in ((POLICY, "    def add_one_req(\n", METHOD),
                                  (SCHED, "            res = adder.add_one_req(\n",
                                   "            if adder.shares_admitted_prefix(req):\n                continue\n")):
        assert f"diff --git a/{path} b/{path}".encode() not in scored
        old = base(path)
        assert old.count(marker) == 1
        new = old.replace(marker, insert + marker)
        compile(new, path, "exec")
        pieces.append("".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/" + path, "b/" + path)))
    (HERE / "delta.diff").write_bytes("".join(pieces).encode())
    print("Compiled guarded admission patch against the scored source.")

if __name__ == "__main__":
    main()

