"""Materialize the serving specialization from the immutable native donor and build for Hopper."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
DONOR = HERE.parent.parent / "muse_moe_w4a16/submission_v139_wgs_kperm3.py"
EXPECTED = "a24b8374aad49fbaa5e0cdeffb7f1fa64960d8321f343cf8b88bd97dfdd20c09"
FLAGS = ["-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
         "-DOKC_KPERM3=1", "-DOKC_PRMTSC=1", "-DOKC_NS=3", "-DOKC_NSF=3", "-DOKC_FNST=4"]

def replace(src, old, new):
    if src.count(old) != 1:
        raise ValueError(f"ambiguous donor anchor {old[:70]}")
    return src.replace(old, new)

def materialize():
    if hashlib.sha256(DONOR.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("donor changed")
    node = next(n for n in ast.parse(DONOR.read_text()).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in n.targets))
    src = ast.literal_eval(node.value).split("#include <torch/extension.h>", 1)[0]
    src = (HERE / "serving_fragment.cuh").read_text() + "\n" + src
    start = src.index("          uint32_t r0, r1, r2, r3;", src.index("void wgs_seg("))
    end = src.index("#endif", start) + len("#endif")
    src = src[:start] + """
          const uint16_t* table = reinterpret_cast<const uint16_t*>(p.lut);
          af[ci & 1][jj][0] = serving_pair<0>(q, scg, table);
          af[ci & 1][jj][1] = serving_pair<1>(q, scg, table);
          af[ci & 1][jj][2] = serving_pair<2>(q, scg, table);
          af[ci & 1][jj][3] = serving_pair<3>(q, scg, table);""" + src[end:]
    epi = src.index("void wgs_epi_tile(")
    end = src.index("void wgs_flush(", epi)
    section = src[epi:end].replace("p.gsA[e] * TWO62","p.gsA[e]")
    a = section.index("const float gt =")
    b = section.index("st_keep_bf16", a)
    section = section[:a] + "const auto v = serving_activation(d4[h], d4[2+h], gsc);\n        " + section[b:]
    section = replace(section, "__float2bfloat16(probe_q(v))", "v")
    a = section.index("#if OKC_WGSFINSS")
    b = section.index("#endif", a) + len("#endif")
    section = section[:a] + """
    const int col = r16 * 16 + g;
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        auto* dst = p.out + (size_t)S.ent[jst + tk] * HD + col;
        st_keep_bf16(dst, __float2bfloat16_rn(d4[h]));
        st_keep_bf16(dst + 8, __float2bfloat16_rn(d4[2+h]));
      }
    }""" + section[b:]
    src = src[:epi] + section + src[end:]
    src = src[:src.index("  // final: out[t]")] + src[src.index("  DBG(5);"):]
    src = src.replace("p.ybuf + (size_t)EMAX * HD + ", "p.ybuf + ")
    src = replace(src, "    float* other = (S.gen & 1) ? p.ss : p.ss1;\n    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;", "")
    src = replace(src, "  wgs_prefix(S, nj);", "  if (nj == 0) { for (int k = 0; k < 4; ++k) S.wgr[k] = 0; return; }\n  wgs_prefix(S, nj);")
    # Assign complete K reductions to one CTA. The original split-K fixups used
    # unordered FP32 atomics and two 10 MiB partial-sum buffers.
    src = replace(src, "  wgs_prefix(S, nj);", "  wgs_prefix(S, nj);")
    start = src.index("    const int slot = (s1 - s0 == 16)")
    src = src[:start] + "    const int slot = -1;\n" + src[src.index("    const int ntt =", start):]
    start = src.index("  int* fc = wgs_cnt(p, PB)")
    src = src[:start] + "}\n" + src[src.index("template <",src.index("if (wt == 0) *fc = 0;", start)):]
    start = src.index("  if (slot >= 0) {  // a share of a unit")
    src = src[:start] + src[src.index("  const int j = u >> UBL",start):]
    anchor = "  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);"
    src = replace(src, anchor, anchor + "\n  for (int k = 0; k < 4; ++k) S.wgr[k] = (S.wgr[k] + 15) & ~15;")
    # Wrap inline-assembly operand lists before emitting source to SGLang.
    lines = []
    for line in src.splitlines():
        while len(line) > 395:
            split = line.rfind("), ", 0, 370)
            if split < 0:
                raise ValueError("unexpected long line in donor")
            lines.append(line[:split+2])
            line = "                  " + line[split+3:]
        lines.append(line.rstrip())
    src = "\n".join(lines) + "\n"

    return src + (HERE / "serving_launch.cuh").read_text()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    source = out / "expert.cu"
    from specialize import serving_source
    source.write_text(serving_source(materialize(),FLAGS))
    library = out / "libexpert.so"
    command = ["nvcc", "-std=c++17", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "--shared", "-Xcompiler=-fPIC",
               "--cudart=shared", "-Xptxas=-v", *FLAGS, str(source), "-o", str(library)]
    with (out / "nvcc.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    (out / "results.json").write_text(json.dumps(dict(donor=EXPECTED, flags=command,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest()), indent=2))

if __name__ == "__main__":
    main()




