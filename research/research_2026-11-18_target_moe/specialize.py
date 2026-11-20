"""Drop donor code for alternate layouts and kernels from the single-layout build."""
import re
import subprocess


def serving_source(source, flags):
    includes = list(dict.fromkeys(re.findall(r"^#include .*$",source,re.M)))
    source = re.sub(r"^#(?:include .*|pragma once)$","",source,flags=re.M)
    # Resolve the fixed Hopper specialization before pruning alternate kernels.
    # CUDA header macros stay untouched until NVCC processes the final source.
    p = subprocess.run(["g++","-E","-P","-C","-undef","-nostdinc","-x","c++","-D__CUDA_ARCH__=900",*flags,"-"],
                       input=source,text=True,capture_output=True,check=True)
    s = p.stdout

    def cut(start,stop):
        nonlocal s
        i=s.index(start)
        j=s.index(stop,i)
        s=s[:i]+s[j:]

    cut("__device__ __forceinline__ uint32_t bmul2(", "__device__ __forceinline__ int ld_acquire")
    # The route fixture and the live kernel use the same grid-index helper.
    cut("// Dequantize one 32-bit word","__device__ void route(")
    s=s.replace("  early_issue(p, S);","")
    cut("// ---------------------------------------------------------------- skinny unit:", "__device__ __forceinline__ uint32_t smem_u32(")
    cut("template <bool PB>\n__device__ __forceinline__ void ring_issue(", "__device__ __forceinline__ uint64_t wg_desc(")
    i=s.index("wgmma.wait_group.sync.aligned")
    i=s.index("}",i)+1
    j=s.index("constexpr int WGS_NS",i)
    s=s[:i]+"\n"+s[j:]
    cut("__device__ __forceinline__ float* wgs_fix(", "constexpr unsigned long long wgs_pack8")
    cut("__device__ __forceinline__ int wgs_cta_of(", "template <bool BIG>\n__global__")
    s=s.replace(" // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered","")
    s=s.replace("  if (0 & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }","")
    cut("  for (int i = tid; i < (1 ? 0 : 256)", "  if (tid == 0) {")
    cut("// e4m3 scale byte b of", "__device__ __forceinline__ void mbar_init")
    s=s.replace("!(0 & 8)", "true")
    s=s.replace("if (!pb && true)", "if (!pb)").replace("if (pb && true)", "if (pb)")
    s=s.replace("  if (0 & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }","")
    s=s.replace("  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;", "")
    s=s.replace("    S.ssp = (gen & 1) ? p.ss1 : p.ss;", "")
    for field in ("preissued", "seq", "fseq", "preissuedB"):
        s=s.replace(f"    S.{field} = 0;", "")
    s=s.replace("    S.xbuf_ptr = xbuf;", "")
    s=s.replace("    S.cnt[lane] = tot;", "").replace("    S.off[lane] = off;", "")
    s=s.replace("      S.pos[i] = (short)q;\n    } else if (i < i1) {\n      S.pos[i] = -1;", "")
    cut("struct Smem {", "// floor(a * c")
    s=s.replace("// floor(a * c", """struct Smem {
  uint64_t bars[24];
  int wcp[JMAX + 1], wcpB[JMAX + 1], jrow[JMAX + 1];
  uint32_t wtot, wtotB;
  int wgr[4], wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], njF, njS, gen;
  signed char eid[EMAX];
  short ent[EMAX];
};

// floor(a * c""")
    s=s.replace("  // programmatic dependent launch: everything below reads the previous kernel's outputs (routing, x) or writes\n  // scratch, so wait for it here; without the launch attribute this returns at once","")
    s=s.replace("  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still\n  // waits for this whole grid to finish and its writes to be visible)","")

    # No profiler, debug clock or grid-dependent launch is used by this entry point.
    lines=[]
    for line in s.splitlines():
        if " = gtimer(); } while (0)" in line or 'asm volatile("griddepcontrol.' in line:
            continue
        while len(line)>395:
            pos=line.rfind("), ",0,375)
            if pos<0: pos=line.rfind(", ",0,375)
            if pos<0: raise ValueError(line)
            pos=line.index(",",pos)+1
            lines.append(line[:pos])
            line="                 "+line[pos:].lstrip()
        lines.append(line.rstrip())
    # The routing kernel uses an external dynamic shared memory declaration.
    s="\n".join(lines)
    s=s.replace("__device__ __forceinline__ uint32_t smem_u32", "constexpr int UNIT_BYTES = 36864;\n__device__ __forceinline__ uint32_t smem_u32")
    if "unsigned char smem_raw[];" not in s:
        s=s.replace("__device__ __forceinline__ int ld_acquire", "extern __shared__ __align__(16) unsigned char smem_raw[];\n__device__ __forceinline__ int ld_acquire")
    return "\n".join(includes)+"\n"+s+"\n"


