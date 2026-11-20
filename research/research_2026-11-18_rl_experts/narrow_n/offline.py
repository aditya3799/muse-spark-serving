"""Build representative 128-column vendor kernels without opening a CUDA context."""
import json,os,sys,hashlib,shutil
from pathlib import Path
from dataclasses import fields
here=Path(__file__).resolve().parent;root=here.parents[1]
sys.path.insert(0,str(root/"swarm_2026-11-16/moe/humming_vendor"))
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
from humming.config import LayerConfig,ComputeConfig,TuningConfig,MmaType
from humming.kernel.humming import HummingKernel
from humming.jit.compiler import NVRTCCompiler
from policy import narrow
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
source=json.loads((old/"occupancy_jobs.json").read_text())["jobs"]
rows=[]
for arch,archstr in ((86,"86"),(90,"90a")):
    for stage in ("w13","w2"):
        item=next(r for r in source if r["stage"]==stage)
        c=dict(item.get("baseline") or item["configuration"])
        c.pop("num_sms",None)
        c.update(use_stream_k=False,use_f16_accum=False,use_warp_spec=False,use_tma=False)
        for k in tuple(c):
            if k.startswith("use_tma_"):c[k]=False
        c.update(block_shape=[16,256,128] if arch==86 else [8,256,128],
                 warp_shape=[16,64,64] if arch==86 else [8,32,64],num_stages=3,num_ctas_per_sm=1)
        c["mma_type"]="mma" if arch==86 else "wgmma"
        assert narrow(c,arch,1024,stage,"both") is not c
        for label,config in (("parent",c),("candidate",narrow(c,arch,1024,stage,"both"))):
            try:
                kernel=Offline(**config)
                path=Path(NVRTCCompiler.compile(kernel.code,sm_version=archstr,kernel_expr=kernel.kernel_expr,disable_fast_math=False))
                dst=here/f"{stage}_{arch}_{label}.cubin";shutil.copyfile(path,dst)
                rows.append(dict(arch=arch,stage=stage,label=label,block=config["block_shape"],warp=config["warp_shape"],
                    cubin_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),threads=kernel.num_threads))
            except Exception as ex:
                rows.append(dict(arch=arch,stage=stage,label=label,error=repr(ex)));print(rows[-1],flush=True)
            (here/"offline.json").write_text(json.dumps(rows,indent=2))
assert not torch.cuda.is_initialized()
print(rows,flush=True)

