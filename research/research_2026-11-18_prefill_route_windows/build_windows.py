from pathlib import Path
import importlib.metadata
import os
import subprocess

D=Path(__file__).resolve().parent
TARGET=D.parent/"research_2026-11-18_target_moe"
PACKAGE=TARGET/"native_package_02"
def once(s,old,new):
    if s.count(old)!=1: raise AssertionError(old)
    return s.replace(old,new)
def build():
    src=(PACKAGE/"hopper_expert.cu").read_text()
    src=once(src,"struct Params {","struct Params {\n  const int* layout; const int* sorted; const int* expert_map; int window, jobs;")
    cut="constexpr int UNIT_BYTES"
    src=once(src,cut,(D/"window_index.cuh").read_text()+"\n"+cut)
    src=src.replace("S.ent[S.jS[j] + r] >> 3","source_slot(p,S,S.jS[j] + r) >> 3").replace("(size_t)S.ent[jst + tk]","(size_t)source_slot(p,S,jst + tk)")
    src=once(src,"moe_fused(Params p) {","moe_fused(Params p) {\n  if (!select_window(p)) return;")
    src=once(src,"  route(p, S);","  route_window(p, S);")
    src=once(src,"for (int li = blockIdx.x * NT + tid; li < p.M","for (int li = blockIdx.x * NT + tid; !p.layout && li < p.M")
    original=src[src.index('extern "C" int expert_run('):src.index('extern "C" size_t expert_dynamic_shared_bytes')]
    extra=original.replace("int expert_run(","int expert_run_window(").replace("int rows, int experts, int grid, cudaStream_t stream","int rows, int experts, int grid, const int* sorted, const int* expert_map, const int* layout, int window, cudaStream_t stream")
    extra=extra.replace(" || rows > MMAX","").replace("  p.wAs =","  p.sorted = sorted; p.expert_map = expert_map; p.layout = layout; p.window = window;\n  p.wAs =")
    src+=extra
    (D/"hopper_window.cu").write_text(src)
    gather=src[src.index("__device__ __forceinline__ uint4 wgs_kperm3_load"):src.index("// Work left from the previous segment")]
    epi=src[src.index("template <bool PB>\n__device__ __forceinline__ void wgs_epi_tile"):src.index("template <bool PB>\n__device__ __forceinline__ void wgs_flush")]
    fixture=(TARGET/"paired_mma.cuh").read_text().replace("  route(p,s);","  route_window(p,s);")
    fixture=fixture.replace("  auto& s =", "  if (!select_window(p)) return;\n  auto& s =")
    fixture=fixture.replace("int experts,cudaStream_t stream)","int experts,const int* sorted,const int* expert_map,const int* layout,int window,cudaStream_t stream)").replace("  Params p{};", "  Params p{}; p.sorted=sorted; p.expert_map=expert_map; p.layout=layout; p.window=window;")
    (D/"paired_windows.cu").write_text(src[:src.index(cut)]+gather+epi+fixture)
    cuda=importlib.metadata.distribution("nvidia-cuda-nvcc")
    nvcc=next(str(cuda.locate_file(x)) for x in cuda.files if str(x).endswith("/bin/nvcc"))
    for name,arch in (("paired_windows","86"),("hopper_window","90a")):
        cmd=[nvcc,"-O3","--std=c++17","-gencode",f"arch=compute_{arch},code=sm_{arch}","-shared","-Xcompiler=-fPIC","--cudart=shared",str(D/(name+".cu")),"-o",str(D/(name+".so"))]
        with (D/(name+"_build.log")).open("w") as f: subprocess.run(cmd,check=True,stdout=f,stderr=f)
if __name__=="__main__": build()
