"""Build the FA3 packet writer on the immutable scored Humming baseline."""
import ast
import hashlib
import re
import subprocess
import tempfile
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sequence_parallel"))
import importlib.util
spec = importlib.util.spec_from_file_location("scored_builder", HERE.parent / "sequence_parallel" / "build.py")
scored = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scored)
BASE, SCORED, PREFIX, REPO, ROOT, diff, replace = (getattr(scored,k) for k in ("BASE","SCORED","PREFIX","REPO","ROOT","diff","replace"))

MODEL = PREFIX + "models/muse_spark.py"
BACKEND = PREFIX + "layers/attention/flashattention_backend.py"
KERNEL = "python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py"

def change(path, src):
    if path == BACKEND:
        src = replace(src, "        self.use_sliding_window_kv_pool = (", '''        self._muse_cast_map = None
        if (fa_impl_ver == 3 and not self.is_draft_runner
                and self.kv_cache_dtype == torch.float8_e4m3fn
                and self.kv_cache_dtype_str != "auto"
                and getattr(model_runner.model_config.hf_config, "model_type", None)
                    in ("muse_spark", "muse_spark_text")):
            # Construct before graph capture with the same running device and
            # torch conversion as the normal FA3 cache writer, including NaNs.
            bits = torch.arange(65536, device=self.device, dtype=torch.int32).to(torch.int16)
            self._muse_cast_map = bits.view(torch.bfloat16).to(self.kv_cache_dtype).view(torch.uint8)

        self.use_sliding_window_kv_pool = (''')
        src = replace(src, "    def get_fused_kv_write_buffers(self, layer, forward_batch):", '''    def muse_fp8_write_buffers(self, layer, forward_batch):
        if (self._muse_cast_map is None or self.topk > 1
                or layer.k_scale is not None or layer.v_scale is not None
                or layer.head_dim > 256 or self.kv_cache_is_mxfp8):
            return None
        result = self._mha_fused_write_buffers(layer, forward_batch)
        if result is None:
            return None
        k, v, loc = result
        if (k.dtype != self.kv_cache_dtype or v.dtype != k.dtype
                or not k.is_contiguous() or not v.is_contiguous()
                or loc.ndim != 1 or not loc.is_contiguous()):
            return None
        return (k.view(k.shape[0], -1), v.view(v.shape[0], -1), loc, self._muse_cast_map)

    def get_fused_kv_write_buffers(self, layer, forward_batch):
        if self.kv_cache_dtype != torch.bfloat16:
            return None
        return self._mha_fused_write_buffers(layer, forward_batch)

    def _mha_fused_write_buffers(self, layer, forward_batch):''')
        return replace(src,"            self.kv_cache_dtype != torch.bfloat16\n            or self.use_mla","            self.use_mla")
    if path == MODEL:
        src = replace(src, "        if fuse_rope:\n            q, k = self._fused_qk_norm(", '''        fa3_store = None
        if (kv_store is None and self._fused_qk_norm is not None
                and not get_memory().enable_unified_memory
                and (forward_batch.forward_mode.is_decode() or forward_batch.forward_mode.is_target_verify())):
            get_write = getattr(get_attn_backend(), "muse_fp8_write_buffers", None)
            if get_write is not None:
                rope_pos = positions if fuse_rope else None
                rope_cache = self.rotary_emb.cos_sin_cache if fuse_rope else None
                if self._fused_qk_norm.can_fuse_fa3(qkv, rope_pos, rope_cache):
                    fa3_store = get_write(self.attn, forward_batch)
                    if fa3_store is not None and fa3_store[2].numel() != qkv.shape[0]:
                        fa3_store = None
        if fa3_store is not None:
            q, k = self._fused_qk_norm._forward_triton(qkv, rope_pos, rope_cache, fa3_store)
        elif fuse_rope:
            q, k = self._fused_qk_norm(''')
        return replace(src, "save_kv_cache=kv_store is None)","save_kv_cache=kv_store is None and fa3_store is None)")
    src = replace(src, "    CACHE_LOC,  # [T] pool row per token","    CACHE_LOC,  # [T] pool row per token\n    CAST_MAP,  # [65536] device Torch BF16 -> E4M3FN byte map for FA3")
    src = replace(src, "    FP8_KV_STORE: tl.constexpr,", "    FP8_KV_STORE: tl.constexpr,\n    FA3_BYTE_STORE: tl.constexpr = False,")
    src = replace(src,"        if FP8_KV_STORE:\n            x = tl.minimum",'''        if FA3_BYTE_STORE:
            bf = tl.where(is_v[:, None], raw, x.to(tl.bfloat16))
            pattern = bf.to(tl.uint16, bitcast=True).to(tl.int32)
            encoded = tl.load(CAST_MAP + pattern)
        elif FP8_KV_STORE:
            x = tl.minimum''')
    src = replace(src, "        # Slot zero is reserved for CUDA-graph padding.\n        kv_mask", '''        if FA3_BYTE_STORE:
            # Match store_cache's invalid physical slot failure and skip slot zero.
            bad = (row_mask & ((loc < 0) | (loc >= kv_capacity))).to(tl.int32)
            tl.inline_asm_elementwise(
                "{ .reg .pred p; setp.ne.s32 p, $1, 0; @p trap; mov.b32 $0, $1; }",
                constraints="=r,r", args=[bad], dtype=tl.int32, is_pure=False, pack=1,
            )
        # Slot zero is reserved for CUDA-graph padding.
        kv_mask''')
    src = replace(src, "        tl.store(\n            K_CACHE",'''        if FA3_BYTE_STORE:
            cache_x = encoded
        else:
            cache_x = x
        tl.store(
            K_CACHE''')
    src=src.replace("kv_dst[:, None] + cols[None, :], x, mask=kv_mask","kv_dst[:, None] + cols[None, :], cache_x, mask=kv_mask")
    src=replace(src, "    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])",'''    if FA3_BYTE_STORE:
        tl.store(Q_OUT + q_dst[:, None] + cols[None, :], encoded, mask=mask & is_q[:, None])
    else:
        tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])''')
    src=replace(src,"    def forward_triton(\n",'''    def can_fuse_fa3(self, qkv, positions, cos_sin_cache):
        from sglang.kernels.fused_op import get_fused_op_backend
        return (
            not self.is_torch_compile and qkv.dtype == torch.bfloat16
            and qkv.is_cuda and not torch.version.hip
            and get_fused_op_backend() in (None, KernelBackend.TRITON)
            and self.auto_selected_backend() == KernelBackend.TRITON
            and self.backend_eligible(KernelBackend.TRITON, qkv, positions, cos_sin_cache)
        )

    def forward_triton(
''')
    src=replace(src,"        has_kv = kv_store is not None\n","        fa3_bytes = kv_store is not None and len(kv_store) == 4\n        has_kv = kv_store is not None\n")
    src=replace(src,"            k_cache, v_cache, cache_loc = kv_store\n            assert", "            k_cache, v_cache, cache_loc = kv_store[:3]\n            assert")
    src=replace(src,"            qkv,\n            q_out,\n            k_out,","            qkv,\n            q_out.view(torch.uint8) if fa3_bytes else q_out,\n            k_out,")
    src=replace(src,"            k_cache if has_kv else qkv,\n            v_cache if has_kv else qkv,\n            cache_loc if has_kv else qkv,","            k_cache.view(torch.uint8) if fa3_bytes else (k_cache if has_kv else qkv),\n            v_cache.view(torch.uint8) if fa3_bytes else (v_cache if has_kv else qkv),\n            cache_loc if has_kv else qkv,\n            kv_store[3] if fa3_bytes else qkv,")
    return replace(src,"            FP8_KV_STORE=fp8_kv,","            FP8_KV_STORE=fp8_kv,\n            FA3_BYTE_STORE=fa3_bytes,")

def main():
    base=(ROOT/"release_2026-11-18/candidate.diff").read_bytes()
    assert hashlib.sha256(base).hexdigest()==SCORED
    items=re.split(r"(?=^diff --git )",base.decode(),flags=re.M)[1:]
    sections={item.splitlines()[0].split()[2][2:]:item for item in items}
    delta=[]
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        subprocess.run(["git","-C",tmp,"init","-q"],check=True)
        for p in (MODEL,BACKEND,KERNEL):
            raw=subprocess.check_output(["git","-C",str(REPO),"show",BASE+":"+p]).decode()
            dest=Path(tmp)/p
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(raw.encode())
            if p in sections:
                subprocess.run(["git","-C",tmp,"apply","-"],input=sections[p].encode(),check=True)
            before=dest.read_bytes().decode()
            after=change(p,before)
            ast.parse(after)
            (HERE/("parent_"+Path(p).name)).write_bytes(before.encode())
            (HERE/Path(p).name).write_bytes(after.encode())
            sections[p]=diff(p,raw,after)
            delta.append(diff(p,before,after))
        subprocess.run(["git","-C",tmp,"apply","--check","-"],input="".join(delta).encode(),check=True)
    (HERE/"delta.diff").write_bytes("".join(delta).encode())
    data="".join(sections.values()).encode()
    (HERE/"candidate.diff").write_bytes(data)
    print(len(data),hashlib.sha256(data).hexdigest())

if __name__=="__main__":
    main()

