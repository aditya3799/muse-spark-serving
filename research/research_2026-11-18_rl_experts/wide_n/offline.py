"""Static resource gate for width 512 with the original warp tile, K and stages."""
import hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
sys.path.insert(0,str(root/"swarm_2026-11-16/moe/humming_vendor"))
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
from humming.config import LayerConfig,ComputeConfig,TuningConfig
from humming.kernel.humming import HummingKernel
from humming.jit.compiler import NVRTCCompiler
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
records=[];resource_exe=shutil.which("cuobjdump") or str(Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13/bin/cuobjdump")
# SM86 includes the real M128/M768 dimensions plus the M8/M32 worklist for a sparse case.
geometry={86:[(16,16,64,64,32,s) for s in (2,3)]+[(16,16,64,128,64,3),(48,48,64,128,64,3)]
            +[(m,w,64,64,64,s) for m,w in ((32,16),(96,48),(128,64)) for s in (2,3)],
          90:[(8,8,32,128,64,4),(16,16,32,128,64,4),(64,64,32,64,64,4),(64,64,32,128,64,4),(96,96,32,64,64,4)]}
for arch,archstr,capacity in ((86,"86",99*1024),(90,"90a",227*1024)):
    for bm,wm,wn,bk,wk,stages in geometry[arch]:
        for stage in ("w13","w2"):
            ref=next(r for r in jobs if r["stage"]==stage)
            c=dict(ref["baseline"]);c.pop("num_sms",None)
            c.update(use_stream_k=False,use_f16_accum=False,use_warp_spec=False,use_tma=False)
            for k in tuple(c):
                if k.startswith("use_tma_"):c[k]=False
            c.update(block_shape=[bm,512,bk],warp_shape=[wm,wn,wk],num_stages=stages,num_ctas_per_sm=1,
                mma_type="mma" if arch==86 else "wgmma")
            item=dict(sm=arch,stage=stage,block=c["block_shape"],warp=c["warp_shape"],stages=stages,
                      capacity_bytes=capacity,allowed=False)
            records.append(item)
            try:
                kernel=Offline(**c)
                smem=kernel.estimated_smem_size;threads=kernel.num_threads
                item.update(threads=threads,dynamic_smem=smem)
                if smem+1024>capacity: item["reason"]="dynamic memory plus conservative 1 KiB static reserve exceeds per-block limit";continue
                if threads>1024: item["reason"]="exceeds CTA thread limit";continue
                code=Path(NVRTCCompiler.compile(kernel.code,sm_version=archstr,kernel_expr=kernel.kernel_expr,disable_fast_math=False))
                dst=here/f"{arch}_{stage}_{bm}_{wm}_{bk}_{wk}_s{stages}.cubin"
                shutil.copyfile(code,dst)
                windows="/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/cuobjdump.exe"
                path=subprocess.check_output(["wslpath","-w",str(dst)]).decode().strip()
                text=subprocess.check_output([windows,"--dump-resource-usage",path]).decode()
                found=re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",text)
                if not found: raise ValueError(text[-800:])
                regs,stack,static,local=map(int,found.groups())
                reg_units=((regs*32+255)//256)*256*(threads//32)
                item.update(registers=regs,registers_per_cta=reg_units,stack=stack,local_bytes=local,static_smem=static,
                    cubin_sha256=hashlib.sha256(code.read_bytes()).hexdigest())
                item["allowed"]=smem+static<=capacity and reg_units<=65536 and not stack and not local
                item["reason"]="admitted" if item["allowed"] else "register budget or spilling"
            except Exception as ex:
                item["reason"]=repr(ex)
            finally:
                (here/"resources_checked.json").write_text(json.dumps(records,indent=2));print(arch,stage,bm,bk,stages,item["allowed"],item.get("reason"),flush=True)
assert not torch.cuda.is_initialized()


