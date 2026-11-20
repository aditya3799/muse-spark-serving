"""Build an UNAPPLIED incremental patch against frozen scored candidate sources."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVING = HERE.parents[1]
BASE = SERVING / 'release_2026-11-18/candidate_sources'
OUT = HERE / 'contiguous_kv_unapplied'
BACKEND = 'python/sglang/srt/layers/attention/flashattention_backend.py'
HELPER = 'python/sglang/srt/layers/attention/contiguous_prefill_kv.py'
KERNEL = 'python/sglang/kernels/ops/kvcache/contiguous_kv_staging.py'

source = (BASE / BACKEND).read_text()
old = '''                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    scheduler_metadata=sched_meta,'''
assert source.count(old) == 1
new = '''                from sglang.srt.layers.attention.contiguous_prefill_kv import maybe_stage

                staged_kv = maybe_stage(
                    self, q, key_cache, value_cache, page_table, cache_seqlens,
                    layer, forward_batch,
                    allowed=(
                        causal and not is_swa_layer and not use_local_attn
                        and not use_cascade_attn and not cp_active
                        and sinks is None and score_mod is None and rel_bias is None
                        and metadata.image_token_ranges is None
                        and metadata.swa_spec_metadata is None
                    ),
                )
                staged_kwargs = {"pack_gqa": True} if staged_kv is not None else {}
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache if staged_kv is None else staged_kv[0],
                    v_cache=value_cache if staged_kv is None else staged_kv[1],
                    page_table=page_table if staged_kv is None else None,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    scheduler_metadata=sched_meta,'''
changed = source.replace(old, new)
# Insert extra kwargs only in the exact modified call, preserving all originals.
start = changed.index(new)
end = changed.index('\n                )', start + len(new))
part = changed[start:end]
assert part.count('                    **kwargs,') == 1
part = part.replace('                    **kwargs,', '                    **staged_kwargs,\n                    **kwargs,')
changed = changed[:start] + part + changed[end:]
kernel = (HERE / 'contiguous_kv.py').read_text()
kernel = kernel[:kernel.index('\n\nclass Arena:')]
kernel = kernel.replace('import threading\n\n', '')
files = {BACKEND: changed, HELPER: (HERE / 'contiguous_prefill_helper.py').read_text(), KERNEL: kernel}
patches = []
manifest = {'status': 'UNAPPLIED; DISABLED DEFAULT; H200 FP8 UNQUALIFIED', 'files': []}
for name, text in files.items():
    ast.parse(text)
    target = OUT / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    prior = source if name == BACKEND else ''
    patches.append(''.join(difflib.unified_diff(prior.splitlines(True), text.splitlines(True),
                    fromfile='a/' + name if prior else '/dev/null', tofile='b/' + name)))
    manifest['files'].append({'path': name, 'sha256': hashlib.sha256(text.encode()).hexdigest()})
patch = ''.join(patches)
(OUT / 'incremental.diff').write_text(patch)
manifest['patch_sha256'] = hashlib.sha256(patch.encode()).hexdigest()
manifest['bytes'] = len(patch.encode())
manifest['base_backend_sha256'] = hashlib.sha256(source.encode()).hexdigest()
manifest['max_added_line'] = max(len(line) for line in patch.splitlines() if line.startswith('+'))
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
