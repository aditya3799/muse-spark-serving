"""Test stock fp8/CTA-K=16 vs two legal K64 instantiations, with a tagged scale type."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import compile_fp8 as comp
import subprocess

inc=Path("/tmp/okc-cutlass-cd9d8a7d/include")
assert (inc/"cutlass/gemm/collective/builders/sm90_sparse_config.inl").is_file()
comp.include=inc
comp.a.update(snapshot="NVIDIA CUTLASS public revision cd9d8a7d612b6b9ad0f1b698925e31243e828629",
              grouped_header_matches_upstream_bytes=True,
              local_header_sha256=hashlib.sha256((inc/comp.header).read_bytes()).hexdigest())
assert comp.a["local_header_sha256"]==comp.a["upstream_header_sha256"]
utils=Path("cutlass/detail/collective/mixed_input_utils.hpp")
base=(inc/utils).read_text()
# A custom packed-scale path exists only in a private build of the typed kernel; every other
# CUTLASS instantiation sees the original branches. Convert each FP4/FP8 weight to BF16 and
# select its exact BF16-normalized per-K16 group scale from the uint32, without FP8 math.
needle="""      if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));"""
assert base.count(needle)==2
new="""      if constexpr (is_same_v<ElementScale, uint32_t> && is_same_v<DstType, cutlass::bfloat16_t>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            const uint32_t word = scales_vm(j, i);
            const uint8_t byte = uint8_t(word >> (8 * k_block));
            const auto scale = cutlass::float_e4m3_t::bitcast(byte);
            const auto normalized_scale = cutlass::bfloat16_t(float(scale) * 0.25f);
            dst_vm(j, i) *= normalized_scale;
          }
        }
      }
      else """+needle.lstrip()
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    overlay=tmp/utils
    overlay.parent.mkdir(parents=True)
    overlay.write_text(base.replace(needle,new,1))
    for k in (16,64):
        comp.compile_one(k)
    comp.compile_one(64,[tmp],["-DPACKED_SCALE"],"custom_words_fp8")
    comp.compile_one(64,[tmp],["-DPACKED_SCALE","-DPACKED_FP4"],"custom_words_fp4")
    patch_lines=subprocess.run(["git","diff","--no-index","--",str(inc/utils),str(overlay)],capture_output=True)
    text=patch_lines.stdout.decode().splitlines()
    first=next(i for i,line in enumerate(text) if line.startswith("@@"))
    (comp.d/"custom_fragment.diff").write_text("--- a/include/"+str(utils)+"\n+++ b/include/"+str(utils)+"\n"+"\n".join(text[first:])+"\n")
