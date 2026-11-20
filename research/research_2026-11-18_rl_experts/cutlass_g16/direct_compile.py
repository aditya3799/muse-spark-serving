"""Compile ordinary indexed Humming on the two architectures with frozen arithmetic and a new loader."""
import os
from pathlib import Path
import json
import hashlib
import subprocess
import sys
import re
here=Path(__file__).resolve().parent;root=here.parents[1]
env_cuda=Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13"
os.environ.update(CUDA_VISIBLE_DEVICES="",CUDA_HOME=str(env_cuda),HUMMING_DEBUG_KERNEL="0",
                  PATH=str(env_cuda/"bin")+":"+os.environ["PATH"])
sys.path.insert(0,str(root/"swarm_2026-11-16/moe/humming_vendor"))
import torch
from humming.config import LayerConfig,ComputeConfig,TuningConfig
from humming.jit.compiler import NVRTCCompiler
from humming.kernel.humming import HummingKernel
from direct_loader import adapted_source
class Offline(HummingKernel):
    def __new__(cls,*a,**kw):return object.__new__(cls)
    def __post_init__(self):
        LayerConfig.__post_init__(self);ComputeConfig.__post_init__(self);TuningConfig.__post_init__(self)
        self.sm_version=arch;self.sm_version_str=archstr;self.init_kernel()
    def prepare(self):assert not torch.cuda.is_initialized()
    def load_cubin(self):raise AssertionError
    @staticmethod
    def current_context():raise AssertionError
old=root/"swarm_2026-11-17/deep_research/hopper_moe"
jobs=json.loads((old/"occupancy_jobs.json").read_text())["jobs"]
vector="--vector" in sys.argv
out=[]
for arch,archstr in ((86,"86"),(90,"90a")):
    for stage in ("w13","w2"):
        c=dict(next(r for r in jobs if r["stage"]==stage)["baseline"])
        c.pop("num_sms",None)
        c.update(use_stream_k=False,use_f16_accum=False,use_warp_spec=False,use_tma=False,use_mbarrier=False)
        for k in tuple(c):
            if k.startswith("use_tma_"):c[k]=False
        if arch==86:
            c.update(block_shape=[16,256,64],warp_shape=[16,64,32],num_stages=2,num_ctas_per_sm=1,mma_type="mma")
        k=Offline(**c)
        for direct in (False,True):
            source=k.code
            if direct:
                needle="#include <humming/kernel/humming.cuh>"
                assert source.count(needle)==1
                source=source.replace(needle,adapted_source(arch==90,vector=vector))
            path=Path(NVRTCCompiler.compile(source,sm_version=archstr,kernel_expr=k.kernel_expr,disable_fast_math=False))
            win=subprocess.check_output(["wslpath","-w",str(path)],text=True).strip()
            info=subprocess.check_output(["/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/cuobjdump.exe","--dump-resource-usage",win],text=True)
            reg,stack,static,local=map(int,re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",info).groups())
            out.append(dict(stage=stage,sm=arch,direct=direct,regs=reg,stack=stack,local_bytes=local,
                            dynamic_smem=k.estimated_smem_size,static_smem=static,bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            (here/("direct_vector_compile.json" if vector else "direct_compile.json")).write_text(json.dumps(out,indent=2))
            print(arch,stage,direct,reg,stack,local,flush=True)
assert not torch.cuda.is_initialized()
