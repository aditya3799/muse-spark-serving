from pathlib import Path
import ast,difflib,hashlib,json
P=Path(__file__).resolve().parent
S=P.parents[3]/'sglang/python/sglang/kernels'
def replace(s, old, new, n=1):
    assert s.count(old)==n, (old[:90],s.count(old))
    return s.replace(old,new)
header=S/'jit/csrc/muse_spark/norm_residual.cuh'
base=header.read_text(); s=base
s=replace(s,'  void* __restrict__ router_out;', '  void* __restrict__ router_out;\n  void* __restrict__ aux_out;\n  int64_t aux_row_stride;')
s=replace(s,'    F32Vec new_stream_v[kFParts];','    F32Vec new_stream_v[kFParts];\n    BF16Vec aux_v;')
s=replace(s,'      new_stream_v[h][l] = new_s;','      new_stream_v[h][l] = new_s;\n      if (params.aux_out != nullptr) aux_v[q] = cast<bf16x2_t>(new_s);')
s=replace(s,'      new_stream_v[j].store(new_stream, idx * kFParts + j);','      new_stream_v[j].store(new_stream, idx * kFParts + j);\n    if (params.aux_out != nullptr) {\n      auto aux = static_cast<bf16_t*>(params.aux_out) + static_cast<int64_t>(blockIdx.x) * params.aux_row_stride;\n      aux_v.store(aux, idx);\n    }')
s=replace(s,'      bool pre_norm_bf16) {','      bool pre_norm_bf16,\n      const tvm::ffi::Optional<tvm::ffi::TensorView> aux_out,\n      int64_t aux_slot) {')
s=replace(s,'    auto G = SymbolicSize{"gate_dim"};','    auto G = SymbolicSize{"gate_dim"};\n    auto W = SymbolicSize{"aux_width"};')
s=replace(s,'    const auto num_rows = static_cast<int>(M.unwrap());','    if (aux_out.has_value()) {\n      TensorMatcher({M, W}).with_dtype<bf16_t>().with_device(device_).verify(aux_out.value());\n      RuntimeCheck(kClusterSize == 1, "aux store requires original single-CTA path");\n      RuntimeCheck(W.unwrap() % kHidden == 0 && aux_slot >= 0 && aux_slot < W.unwrap() / kHidden, "invalid aux feature slot");\n    }\n    const auto num_rows = static_cast<int>(M.unwrap());')
s=replace(s,'        .router_out = kHasRouter ? router_out.value().data_ptr() : nullptr,','        .router_out = kHasRouter ? router_out.value().data_ptr() : nullptr,\n        .aux_out = aux_out.has_value() ? static_cast<bf16_t*>(aux_out.value().data_ptr()) + aux_slot * kHidden : nullptr,\n        .aux_row_stride = aux_out.has_value() ? W.unwrap() : 0,')
(P/'norm_residual_aux.cuh').write_text(s)
wrapper=S/'ops/layernorm/muse_spark_norm_residual_jit.py';wb=wrapper.read_text();w=wb
w=replace(w,'from typing import TYPE_CHECKING, Optional, Tuple','from typing import TYPE_CHECKING, Optional, Tuple\nfrom pathlib import Path')
w=replace(w,'"muse_spark_norm_residual",','"muse_spark_norm_residual_aux_research",')
w=replace(w,'cuda_files=["muse_spark/norm_residual.cuh"],','cuda_files=[str(Path(__file__).with_name("norm_residual_aux.cuh"))],')
w=replace(w,'    cluster_size: Optional[int] = None,','    cluster_size: Optional[int] = None,\n    aux_out: Optional[torch.Tensor] = None,\n    aux_slot: int = 0,')
w=replace(w,'    m, hidden_size = branch.shape','    m, hidden_size = branch.shape\n    if aux_out is not None:\n        if (aux_out.dtype != torch.bfloat16 or aux_out.device != branch.device\n            or aux_out.ndim != 2 or aux_out.shape[0] != m\n            or not aux_out.is_contiguous() or aux_out.shape[1] % hidden_size\n            or type(aux_slot) is not int or not 0 <= aux_slot < aux_out.shape[1] // hidden_size):\n            raise ValueError("invalid auxiliary output owner or slot")')
w=replace(w,'        bool(pre_norm_bf16),','        bool(pre_norm_bf16),\n        aux_out if cluster_size == 1 else None,\n        aux_slot,')
w=replace(w,'    return output, new_stream_out, router_out\n', '    return output, new_stream_out, router_out\n',n=2)
# Only the final return follows a launch; zero-row return stays unchanged.
old='    )\n    return output, new_stream_out, router_out\n'
new='    )\n    if aux_out is not None and cluster_size != 1:\n        aux_out[:, aux_slot * hidden_size : (aux_slot + 1) * hidden_size].copy_(new_stream_out)\n    return output, new_stream_out, router_out\n'
w=replace(w,old,new)
ast.parse(w);(P/'norm_residual_aux.py').write_text(w)
diff=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile=str(header),tofile='norm_residual_aux.cuh'))
diff+=''.join(difflib.unified_diff(wb.splitlines(True),w.splitlines(True),fromfile=str(wrapper),tofile='norm_residual_aux.py'))
(P/'primitive.diff').write_text(diff)
(P/'manifest.json').write_text(json.dumps({'scope':'Separate primitive only; no model integration or compilation yet','inputs':{str(header):hashlib.sha256(base.encode()).hexdigest(),str(wrapper):hashlib.sha256(wb.encode()).hexdigest()},'outputs':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['norm_residual_aux.cuh','norm_residual_aux.py','primitive.diff']}},indent=2)+'\n')
print('Prepared isolated auxiliary store primitive')
