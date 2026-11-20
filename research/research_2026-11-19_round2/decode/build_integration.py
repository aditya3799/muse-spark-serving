"""Build unapplied prologue delta and cumulative candidate over frozen release."""
import ast,difflib,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PIN='297e54fd105e974cdf401e5b4d20253a22c16f90'
BASE='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'
def parent(path):
 p=ROOT/'release_2026-11-18/candidate_sources'/path
 return p.read_text() if p.exists() else subprocess.check_output(['git','-C',str(ROOT/'sglang'),'show',PIN+':'+path]).decode()
def once(text,before,after):
 assert text.count(before)==1,(before[:100],text.count(before))
 return text.replace(before,after)
def main():
 frozen=(ROOT/'release_2026-11-18/candidate.diff').read_bytes();assert hashlib.sha256(frozen).hexdigest()==BASE
 model_path='python/sglang/srt/models/muse_spark.py';backend_path='python/sglang/srt/layers/attention/flashattention_backend.py'
 model=parent(model_path);backend=parent(backend_path)
 backend=once(backend,'    def get_fused_kv_write_buffers(self, layer, forward_batch):',
  '    def get_fused_kv_write_buffers(self, layer, forward_batch, *, allow_fp8_prologue=False):')
 backend=once(backend,'            self.kv_cache_dtype != torch.bfloat16\n            or self.use_mla',
 '''            not (self.kv_cache_dtype == torch.bfloat16 or (
                allow_fp8_prologue and self.kv_cache_dtype == torch.float8_e4m3fn
                and self.fa_impl_ver == 3 and layer.k_scale is None and layer.v_scale is None
                and layer.head_dim == layer.v_head_dim
            ))
            or self.use_mla''')
 model=once(model,'from sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store import FusedQKNorm',
  'from sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store import FusedQKNorm\nfrom sglang.kernels.ops.attention.muse_fa3_prologue.host import maybe_fuse as _maybe_fuse_fa3_prologue')
 begin=model.index('        fuse_rope = self._fused_qk_norm is not None and self.rotary_emb is not None')
 end=model.index('        output = self._fused_o_norm_gate(output, o_gate)',begin)
 native=model[begin:end]
 branch='''        fused_q = _maybe_fuse_fa3_prologue(self, qkv, positions, forward_batch)
        if fused_q is not None:
            output = self.attn(fused_q, None, None, forward_batch, save_kv_cache=False)
        else:
'''+''.join('    '+line if line.strip() else line for line in native.splitlines(keepends=True))
 model=model[:begin]+branch+model[end:]
 changed={model_path:model,backend_path:backend}
 added={'python/sglang/kernels/ops/attention/muse_fa3_prologue/'+p.name:p.read_text() for p in (HERE/'prologue_package').glob('*.py')}
 dest=HERE/'integration';dest.mkdir(exist_ok=True);delta=[];manifest=[]
 for path,text in {**changed,**added}.items():
  ast.parse(text);output=dest/'sources'/path;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text)
  before=parent(path) if path in changed else ''
  delta.extend(difflib.unified_diff(before.splitlines(keepends=True),text.splitlines(keepends=True),fromfile='a/'+path if before else '/dev/null',tofile='b/'+path))
  manifest.append(dict(path=path,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(before.encode()).hexdigest() if before else None))
 diff=''.join(delta);(dest/'delta.diff').write_text(diff)
 cumulative=frozen+b'\n'+diff.encode();(dest/'candidate.diff').write_bytes(cumulative)
 (dest/'manifest.json').write_text(json.dumps(dict(parent=BASE,pin=PIN,applied=False,enabled_by_default=False,
  candidate_sha256=hashlib.sha256(cumulative).hexdigest(),files=manifest),indent=2))
 print('prepared unapplied',dest,'bytes',len(cumulative))
if __name__=='__main__':main()
