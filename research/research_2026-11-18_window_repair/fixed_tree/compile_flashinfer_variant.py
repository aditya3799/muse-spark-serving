"""Compile the tagged wheel's actual Hopper paged FP8 kernels with the custom tree variant."""
import argparse,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
from jinja2 import Template
from flashinfer_variant import DECL,TENSORS,TD,SCALARS,SD

H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("wheel",type=Path);p.add_argument("--width",type=int,default=8)
    a=p.parse_args(); assert sha(a.wheel)=="770550a0aa92b2f10e6c9a1f048dd43713f0f3bb6a12710186fcf8e4e9c99bd8"
    dest=Path("/tmp/okc_flashinfer_tree_072")
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(a.wheel) as z:
        prefixes=("flashinfer/data/include/","flashinfer/data/cutlass/include/","flashinfer/data/cutlass/tools/util/include/","flashinfer/data/csrc/")
        for f in z.namelist():
            if f.startswith(prefixes) and not (dest/f).exists():z.extract(f,dest)
    data=dest/"flashinfer/data";nvcc=next(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cu13/bin/nvcc"))
    attrs="\n".join([f"{typ}* {n};" for n,typ in zip(TENSORS,TD)]+[f"{typ} {n};" for n,typ in zip(SCALARS,SD)])
    result=[]
    for sliding in (False,True):
        name=f"flashinfer::TreeFP8Attention<{a.width},{str(sliding).lower()}>"
        out=dest/f"w{a.width}_s{int(sliding)}";out.mkdir(exist_ok=True)
        inc=Template((data/"csrc/batch_prefill_sm90_customize_config.jinja").read_text()).render(
            dtype_q="__nv_fp8_e4m3",dtype_kv="__nv_fp8_e4m3",dtype_o="__nv_bfloat16",idtype="int32_t",
            head_dim_qk=64,head_dim_vo=64,use_logits_soft_cap="false",use_sliding_window="false",
            additional_params_decl=attrs,additional_func_params="",additional_params_setter="",variant_decl=DECL,variant_name=name)
        (out/"batch_prefill_sm90_config.inc").write_text(inc)
        cu=Template((data/"csrc/batch_prefill_fp8_paged_sm90_kernel_inst.jinja").read_text()).render(
            head_dim_qk=64,mask_mode="MaskMode::kCausal",use_sliding_window="false",variant_name=name)
        unit=out/"native.cu";unit.write_text(cu);bin=out/"native.cubin"
        command=[str(nvcc),"-std=c++17","-arch=sm_90a","-O3","--expt-relaxed-constexpr","--extended-lambda",
                 "-I"+str(data/"include"),"-I"+str(data/"cutlass/include"),"-I"+str(data/"cutlass/tools/util/include"),
                 "-I"+str(out),"--cubin",str(unit),"-o",str(bin)]
        proc=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out/"nvcc.log").write_text(proc.stdout)
        if proc.returncode:
            print(proc.stdout[-4500:]);return proc.returncode
        result.append(dict(sliding=sliding,cubin_bytes=bin.stat().st_size,sha256=sha(bin),flags=command[1:8],
            native_dispatch_variants=2))
        print(result[-1],flush=True)
    report=dict(wheel_sha256=sha(a.wheel),variant_sha256=hashlib.sha256(DECL.encode()).hexdigest(),width=a.width,
                compile_scope="Actual wheel's Hopper paged FP8 attention, both same-schedule modes, no runtime or GPU.",kernels=result)
    (H/f"flashinfer_sm90_w{a.width}.json").write_text(json.dumps(report,indent=2))
    return 0
if __name__=="__main__":sys.exit(main())

