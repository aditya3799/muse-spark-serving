"""Layer fusion overlay layered on verified direct-copy owner; research only."""
from pathlib import Path
import ast,difflib,hashlib,json
P=Path(__file__).resolve().parent;A=P.parent;ROOT=P.parents[3]
OUT=P/'integration_sources';BASE=A/'candidate_sources';records=[];patch=[]
def replace(s,old,new,n=1):
    assert s.count(old)==n,(old[:80],s.count(old))
    return s.replace(old,new)
def save(rel,base,s):
    if rel.endswith('.py'):ast.parse(s)
    dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(s)
    patch.extend(difflib.unified_diff((base or '').splitlines(True),s.splitlines(True),fromfile='a/'+rel if base else '/dev/null',tofile='b/'+rel))
    records.append({'path':rel,'base_sha256':hashlib.sha256(base.encode()).hexdigest() if base else None,'sha256':hashlib.sha256(s.encode()).hexdigest()})
# Preserve all owner changes, replacing only helper/model below.
for path in BASE.rglob('*.py'):
    rel=path.relative_to(BASE).as_posix();s=path.read_text();base=s
    if rel.endswith('/prefill_aux_output.py'):
        s=replace(s,'    def capture(self, layer_idx, residual):','    def capture(self, layer_idx, residual, already_written=False):')
        s=replace(s,'        self.buffer[:, start : start + self.hidden_size].copy_(residual)','        if not already_written:\n            self.buffer[:, start : start + self.hidden_size].copy_(residual)')
    if rel.endswith('/muse_spark.py'):
        before='        forward_batch: ForwardBatch,\n    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:\n        """Returns the updated residual stream and the next layer\'s attention input."""'
        after=before.replace('        forward_batch: ForwardBatch,','        forward_batch: ForwardBatch,\n        aux_out: Optional[torch.Tensor] = None,\n        aux_slot: int = 0,')
        s=replace(s,before,after)
        s=replace(s,'            return stream, next_attn_input\n        if self._fused_ffn_boundary is None:', '            if aux_out is not None:\n                width = stream.shape[-1]\n                aux_out[:, aux_slot * width : (aux_slot + 1) * width].copy_(stream)\n            return stream, next_attn_input\n        if self._fused_ffn_boundary is None:')
        s=replace(s,'            return self.post_feedforward_residual_gate(\n                stream, self.post_feedforward_layernorm(ffn_output)\n            ), None','            stream = self.post_feedforward_residual_gate(\n                stream, self.post_feedforward_layernorm(ffn_output)\n            )\n            if aux_out is not None:\n                width = stream.shape[-1]\n                aux_out[:, aux_slot * width : (aux_slot + 1) * width].copy_(stream)\n            return stream, None')
        s=replace(s,'            stream, ffn_output, branch_dtype=self.operator_dtype\n','            stream, ffn_output, branch_dtype=self.operator_dtype,\n            aux_out=aux_out, aux_slot=aux_slot,\n')
        s=replace(s,'            stream, attn_input = layer(positions, stream, attn_input, forward_batch)','            owns_tap = packed_aux is not None and layer_idx in self.layers_to_capture\n            if owns_tap:\n                stream, attn_input = layer(\n                    positions, stream, attn_input, forward_batch,\n                    aux_out=packed_aux.buffer, aux_slot=packed_aux.index,\n                )\n            else:\n                stream, attn_input = layer(positions, stream, attn_input, forward_batch)')
        s=replace(s,'                    packed_aux.capture(layer_idx, stream)','                    packed_aux.capture(layer_idx, stream, already_written=True)')
    save(rel,base,s)
rel='python/sglang/kernels/ops/layernorm/muse_spark_norm_residual.py';base=(ROOT/'sglang'/rel).read_text();s=base
s=replace(s,'        pre_norm_dtype: torch.dtype | None = None,','        pre_norm_dtype: torch.dtype | None = None,\n        aux_out: torch.Tensor | None = None,\n        aux_slot: int = 0,',n=4)
# Preserve exact backend eligibility: auxiliary destination does not change the selected arithmetic backend.
s=replace(s,'        return output, new_stream, router_input','        if aux_out is not None:\n            width = new_stream.shape[-1]\n            aux_out[:, aux_slot * width : (aux_slot + 1) * width].copy_(new_stream)\n        return output, new_stream, router_input')
s=replace(s,'                pre_norm_dtype=pre_norm_dtype,','                pre_norm_dtype=pre_norm_dtype,\n                aux_out=aux_out, aux_slot=aux_slot,',n=3)
s=replace(s,'        return (\n            output,\n            new_stream_out,','        if aux_out is not None:\n            aux_out[:, aux_slot * H : (aux_slot + 1) * H].copy_(new_stream_out)\n        return (\n            output,\n            new_stream_out,')
old='        from sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit import (\n            fused_norm_residual as _jit_fused_norm_residual,\n        )\n\n        return _jit_fused_norm_residual('
new='        if aux_out is not None:\n            from sglang.kernels.ops.layernorm.muse_spark_norm_residual_aux_jit import (\n                fused_norm_residual as _jit_aux_norm_residual,\n            )\n            return _jit_aux_norm_residual(\n                branch, stream, **self.jit_kernel_kwargs(branch_dtype),\n                want_router=router_dtype is not None,\n                pre_norm_bf16=pre_norm_dtype is torch.bfloat16,\n                aux_out=aux_out, aux_slot=aux_slot,\n            )\n\n'+old
s=replace(s,old,new);save(rel,base,s)
rel='python/sglang/kernels/ops/layernorm/muse_spark_norm_residual_aux_jit.py'
s=(P/'norm_residual_aux.py').read_text().replace('from pathlib import Path\n','').replace('cuda_files=[str(Path(__file__).with_name("norm_residual_aux.cuh"))],','cuda_files=["muse_spark/norm_residual_aux.cuh"],')
save(rel,None,s)
save('python/sglang/kernels/jit/csrc/muse_spark/norm_residual_aux.cuh',None,(P/'norm_residual_aux.cuh').read_text())
(P/'integration_delta.diff').write_text(''.join(patch))
(P/'integration_manifest.json').write_text(json.dumps({'status':'UNVALIDATED_OPTIONAL_FUSION_OVERLAY','base':'Parent candidate_sources verified direct-copy owner; this is a delta, not the full candidate','files':records},indent=2)+'\n')
print('Prepared optional fusion integration overlay')
