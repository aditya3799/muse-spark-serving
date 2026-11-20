from pathlib import Path
import hashlib
import subprocess
import sysconfig
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(path,digest):
    b=(ROOT/path).read_bytes();assert hashlib.sha256(b).hexdigest()==digest
    return b.decode()
base=read("research_2026-11-18_rl/lt_probe.cc","caaeaf026ec51bd5eb94c356ed770931a1a2f68bfbdfc67e24e3aa1270440fe1")
sweep=read("research_2026-11-18_lt_audit/sweep.cc","8832d7b0627a7c4f0a662034dfd2fe1fe7ffef320b89a3c3ec2a55c81cd000d3")
def repl(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
base=repl(base,"input{}, output{}", "input{}, parent_input{}, output{}")
base=repl(base,"{weight, input, output, parent_weight, parent_output}","{weight, input, parent_input, output, parent_weight, parent_output}")
base=repl(base,"cublasOperation_t trans = CUBLAS_OP_T","cublasOperation_t trans = CUBLAS_OP_N")
base=repl(base,"&p->weight, CUDA_R_16BF, k, n, k","&p->weight, CUDA_R_16BF, n, k, n")
base=repl(base,"&p->parent_weight, CUDA_R_16BF, k, full_n, k","&p->parent_weight, CUDA_R_16BF, n, k, n")
base=repl(base,"    CHECK(cublasLtMatrixLayoutCreate(&p->output, CUDA_R_16BF, n, m, n));","    CHECK(cublasLtMatrixLayoutCreate(&p->parent_input, CUDA_R_16BF, k, full_n, k));\n    CHECK(cublasLtMatrixLayoutCreate(&p->output, CUDA_R_32F, n, m, n));")
base=repl(base,"&p->parent_output, CUDA_R_16BF, full_n, m, full_n","&p->parent_output, CUDA_R_32F, n, full_n, n")
base=repl(base,"p->op,w,p->input,y,y,p->pref,32","p->op,w,full?p->parent_input:p->input,y,y,p->pref,32")
sweep=repl(sweep,"p->op,w,p->input,y,y,&algo,out","p->op,w,parent?p->parent_input:p->input,y,y,&algo,out")
sweep=repl(sweep,"&p->output,CUDA_R_16BF,n,m,ldout","&p->output,CUDA_R_32F,n,m,ldout")
sweep=repl(sweep,"p->op,w,p->input,y,y,p->pref,64","p->op,w,full?p->parent_input:p->input,y,y,p->pref,64")
sweep=repl(sweep,"w,weight,x,p->input,&beta","w,weight,x,full?p->parent_input:p->input,&beta")
sweep=repl(sweep,'#include "../research_2026-11-18_rl/lt_probe.cc"',base)
start=sweep.index("    std::vector<uint64_t> workspaces")
end=sweep.index("    for(auto bytes:workspaces)",start)
sweep=sweep[:start]+"    std::vector<uint64_t> workspaces{native_ws};\n"+sweep[end:]
out=HERE/"row_sweep.cc";out.write_bytes(sweep.encode())
toolkit=Path(sysconfig.get_paths()["purelib"])/"nvidia/cu13"
subprocess.run(["g++","-O2","-std=c++17","-fPIC","-shared",str(out),"-I"+str(toolkit/"include"),"-I"+str(toolkit/"include/cccl"),
                "-L"+str(toolkit/"lib"),"-Wl,-rpath,"+str(toolkit/"lib"),"-l:libcublasLt.so.13","-o",str(HERE/"row_sweep.so")],check=True)

