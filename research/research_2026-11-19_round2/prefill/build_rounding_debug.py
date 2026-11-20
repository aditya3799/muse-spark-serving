"""Generate local diagnostic copies; never edits frozen kernels."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1];D=H.parent/'decode'
out=H/'prologue_rounding_debug';out.mkdir(exist_ok=True)
paths={'native':R/'sglang/python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py','candidate':D/'prologue_package/kernel.py'}
manifest={}
for kind,path in paths.items():
 s=path.read_text()
 if kind=='native':
  s='import triton\nimport triton.language as tl\nFP8_E4M3_MAX=448.0\n\n'+s[s.index('@triton.jit\ndef _qk_rms_norm_fwd_kernel'):s.index('\ndef _triton_config')]
  s=s.replace('_qk_rms_norm_fwd_kernel','native_debug_kernel',1)
 else:s=s.replace('def fusion18_kernel(','def candidate_debug_kernel(',1)
 anchor='    eps,\n';assert s.count(anchor)==1
 s=s.replace(anchor,'    DBG_X, DBG_VAR, DBG_RSTD, DBG_BF16,\n'+anchor)
 anchor='    if FP8_KV_STORE:\n        x = x.to(QKV.dtype.element_ty).to(tl.float32)'
 assert s.count(anchor)==1
 insert='''    dbg_dst = (token * NUM_KV_HEADS + head - NUM_Q_HEADS) * D
    dbg_mask = mask & is_k[:, None]
    tl.store(DBG_X + dbg_dst[:, None] + cols[None, :], x, mask=dbg_mask)
    tl.store(DBG_BF16 + dbg_dst[:, None] + cols[None, :], x, mask=dbg_mask)
    tl.store(DBG_VAR + token * NUM_KV_HEADS + head - NUM_Q_HEADS, var, mask=row_mask & is_k)
    tl.store(DBG_RSTD + token * NUM_KV_HEADS + head - NUM_Q_HEADS, rstd, mask=row_mask & is_k)
'''
 s=s.replace(anchor,insert+anchor)
 dest=out/(kind+'_debug.py');dest.write_text(s)
 manifest[kind]=dict(original=str(path),original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),diagnostic_sha256=hashlib.sha256(s.encode()).hexdigest())
(out/'source_manifest.json').write_text(json.dumps(dict(scope='Instrumented clones add stores; must compare final outputs against original kernels to detect observer perturbation.',files=manifest),indent=2))
print('prepared instrumentation only')
