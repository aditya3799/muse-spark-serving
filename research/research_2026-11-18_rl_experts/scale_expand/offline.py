"""Compile exact pinned SM90a indexed Humming kernels with alternate scale dtype."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
src=(here.parent/"narrow_n/offline.py").read_text()
setup=src[:src.index("from policy import narrow")]
klass=src[src.index("class Offline"):src.index('old=root/')]
exec(setup+klass,globals())
arch,archstr=90,"90a"
jobs=json.loads((root/"swarm_2026-11-17/deep_research/hopper_moe/occupancy_jobs.json").read_text())["jobs"]

res=[]
for stage in ("w13","w2"):
    config=next(job["baseline"] for job in jobs if job["stage"]==stage)
    for dtype in ("float8e4m3","bfloat16"):
        c=dict(config,bs_dtype=dtype)
        c.pop("num_sms",None)
        kernel=Offline(**c)
        path=Path(NVRTCCompiler.compile(kernel.code,sm_version=archstr,kernel_expr=kernel.kernel_expr,disable_fast_math=False))
        dst=here/f"{stage}_{dtype}_sm90.cubin";shutil.copyfile(path,dst)
        res.append(dict(stage=stage,bs_dtype=dtype,block=kernel.block_shape,warp=kernel.warp_shape,stages=kernel.num_stages,
            sm=archstr,dynamic_smem=kernel.estimated_smem_size,threads=kernel.num_threads,sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
        (here/"offline.json").write_text(json.dumps(res,indent=2))
assert not torch.cuda.is_initialized()
