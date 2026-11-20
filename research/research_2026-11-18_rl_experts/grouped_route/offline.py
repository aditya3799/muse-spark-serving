"""Compile the pinned Hopper grouped matrices at the exact target fixture sizes."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
src=(here.parent/"narrow_n/offline.py").read_text()
exec(src[:src.index("from policy import narrow")]+src[src.index("class Offline"):src.index("old=root/")],globals())
arch,archstr=90,"90a"
inputs=root/"swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json"
stages=json.loads(inputs.read_text())["stages"]
rows=[]
for m in (128,768):
    for stage,doc in stages.items():
        get=lambda size:next(cfg for lo,hi,cfg in doc["intervals"] if lo<size<=hi)
        heavy=get(m*8)
        for part,selected in (("light",get(m)),("heavy",heavy)):
            config={**doc["meta"],**selected,"gemm_type":"grouped_contiguous","use_cp_async":True,"use_f16_accum":False,"use_stream_k":False}
            config["block_shape"]=list(config["block_shape"]);config["warp_shape"]=list(config["warp_shape"])
            if m*8>4096 and part=="light":
                config["block_shape"][2]=heavy["block_shape"][2]
                config["warp_shape"][2]=heavy["warp_shape"][2]
            row=dict(rows=m,part=part,stage=stage,sm=arch,block=config["block_shape"],warp=config["warp_shape"],stages=config["num_stages"],allowed=False)
            try:
                kernel=Offline(**config)
                row.update(threads=kernel.num_threads,dynamic_smem=kernel.estimated_smem_size)
                binary=Path(NVRTCCompiler.compile(kernel.code,sm_version=archstr,kernel_expr=kernel.kernel_expr,disable_fast_math=False))
                dst=here/f"grouped_{m}_{part}_{stage}_sm90.cubin";shutil.copyfile(binary,dst)
                wpath=subprocess.check_output(["wslpath","-w",str(dst)]).decode().strip()
                exe="/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/cuobjdump.exe"
                usage=subprocess.check_output([exe,"--dump-resource-usage",wpath]).decode()
                row["resources"]={k:int(v) for k,v in re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)",usage)}
                res=row["resources"]
                reg=((res["REG"]*32+255)//256)*256*(kernel.num_threads//32)
                row.update(registers_per_cta=reg,sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
                row["allowed"]=not res["LOCAL"] and not res["STACK"] and kernel.num_threads<=1024 and reg<=65536 and row["dynamic_smem"]+res["SHARED"]<=227*1024
            except Exception as ex:
                row["error"]=repr(ex)
            rows.append(row)
            (here/"hopper_groups.json").write_text(json.dumps(rows,indent=2));print(stage,m,part,row["allowed"],row.get("error",""),flush=True)
assert not torch.cuda.is_initialized()
