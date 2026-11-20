"""Create a Python-only delta that splits oversized host SWA windows before restore."""
from pathlib import Path
import ast,difflib,hashlib,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NAME="python/sglang/srt/mem_cache/unified_radix_cache.py"
BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
SCORED="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
parent=(ROOT/"release_2026-11-18/candidate.diff").read_bytes()
assert hashlib.sha256(parent).hexdigest()==SCORED and ("diff --git a/"+NAME+" ").encode() not in parent
src=subprocess.check_output(["git","-C",str(ROOT/"sglang"),"show",BASE+":"+NAME]).decode()
marker="        self._apply_cache_actions(result.cache_actions)\n        return self._finalize_match_result(params, result)"
new='''        self._apply_cache_actions(result.cache_actions)
        window = self._sliding_window_size
        if (
            window and result.swa_host_hit_length > window + self.page_size - 1
            and self.cache_controller is not None and not self.is_mamba_enabled
            and self.linker is None and not params.key.is_bigram
        ):
            # Restore complete, page-aligned nodes while keeping the original host
            # indices for older branches. Both native tree cores split an edge when
            # a prefix match ends inside it, including any pending transfer receipt.
            end = result.device_prefix_len + result.host_hit_length
            start = (end - window) // self.page_size * self.page_size
            if 0 < start < len(params.key):
                key = RadixKey(params.key.token_ids, extra_key=params.key.extra_key,
                               cache_salt=params.key.cache_salt, limit=start)
                boundary = self.tree_core.match_prefix(MatchPrefixParams(key=key))
                self._apply_cache_actions(boundary.cache_actions)
                result = self.tree_core.match_prefix(params)
                self._apply_cache_actions(result.cache_actions)
        return self._finalize_match_result(params, result)'''
assert src.count(marker)==1
changed=src.replace(marker,new)
ast.parse(changed)
delta=("diff --git a/{0} b/{0}\n".format(NAME)+"".join(difflib.unified_diff(src.splitlines(True),changed.splitlines(True),"a/"+NAME,"b/"+NAME))).encode()
(HERE/"delta.diff").write_bytes(delta)
(HERE/"unified_radix_cache.py").write_bytes(changed.encode())
subprocess.run(["git","-C",str(ROOT/"sglang"),"apply","--check","-"],input=delta,check=True)
print("Source and combined-parent compatibility checked.")

