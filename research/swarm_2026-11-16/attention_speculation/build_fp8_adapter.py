from pathlib import Path
import ast, hashlib, json, textwrap

out=Path(__file__).parent/'fp8_validation'
root=Path(__file__).resolve().parent.parent.parent
baseline_path=root.parent/'muse_attention_decode/baseline.py'
baseline=baseline_path.read_text(encoding='utf-8')
candidate_path=Path(__file__).parent/'candidate/python/sglang/srt/layers/attention/flashattention_backend.py'
candidate=candidate_path.read_text(encoding='utf-8')
cls=next(n for n in ast.parse(candidate).body if isinstance(n,ast.ClassDef) and n.name=='FlashAttentionBackend')
methods=[]
for name in ('_compute_scheduler_metadata','_compute_verify_scheduler_metadata'):
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    methods.append(''.join(candidate.splitlines(True)[node.lineno-1:node.end_lineno]))
helper='class VerifyScheduler:\n'+ '\n'.join(methods)+'\n\n'
helper+='''def _verify_scheduler(data, cfg):
    # Compute from this call's live device lengths. Nothing derived from data
    # survives in a Python cache; the calculation is part of kernel_fn.
    adapter = VerifyScheduler()
    adapter._get_scheduler_metadata = get_scheduler_metadata
    adapter.use_mla = False
    adapter._disable_scheduler_metadata_precompute = False
    adapter.local_attn_builder = None
    adapter._kv_shard_pool = None
    adapter.topk = 1
    adapter.num_attention_heads = cfg["num_q_heads"]
    adapter.num_kv_heads = cfg["num_kv_heads"]
    adapter.head_dim = cfg["head_dim"]
    adapter.kv_cache_dtype = FP8
    adapter.page_size = cfg["page_size"]
    adapter.has_softcap = False
    adapter.num_splits = 0
    metadata = SimpleNamespace(
        page_table=data["page_table"],
        cache_seqlens_int32=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seq_len_q=cfg["q_len"],
    )
    return adapter._compute_verify_scheduler_metadata(metadata, cfg["batch"])


'''
source='# okc: problem=muse_attention_decode\n'
source+='# Full-semantic adapter based on the provided SGLang task baseline.\n'
source+='# Purpose: validate serving scheduler reuse geometry on Hopper FP8.\n'
source+='# One layer cannot demonstrate the across-layer reuse speedup.\n'
source+=baseline
source=source.replace('from typing import TypedDict','from typing import TypedDict\nfrom types import SimpleNamespace')
source=source.replace('from sgl_kernel.flash_attn import flash_attn_with_kvcache','from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata')
source=source.replace('def kernel_fn(data):',helper+'def kernel_fn(data):')
source=source.replace('    out = flash_attn_with_kvcache(','    scheduler = data["scheduler_metadata"]\n    if not window and length > 1:\n        scheduler = _verify_scheduler(data, cfg)\n    out = flash_attn_with_kvcache(')
source=source.replace('scheduler_metadata=data["scheduler_metadata"],','scheduler_metadata=scheduler,')
ast.parse(source)
out.mkdir(exist_ok=True)
(out/'submission.py').write_bytes(source.encode())
manifest=dict(status='prepared_unsubmitted',purpose='Full-semantic task validation of exact serving FA3 verify scheduler helper on Hopper FP8; not an across-layer performance estimate',adapter_sha256=hashlib.sha256(source.encode()).hexdigest(),adapter_bytes=len(source.encode()),serving_candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256('\n'.join(methods).encode()).hexdigest(),baseline_sha256=hashlib.sha256(baseline.encode()).hexdigest(),baseline_origin=str(baseline_path),baseline_commit='a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f',serving_pinned_commit='297e54fd105e974cdf401e5b4d20253a22c16f90',notes=['All task QK-norm/RoPE/cache writes/attention/gate operations preserved','Scheduler computed inside every kernel_fn call for full attention q_len>1','No reference or harness imports; no task IO or diagnostics','Kernel baseline is older than serving pinned release; validates exact helper/API geometry, not entire serving integration'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
