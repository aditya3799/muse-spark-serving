"""Compile actual indexed W13 and W2 configurations without attaching to a GPU."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
cuda=Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13"
os.environ.update(CUDA_VISIBLE_DEVICES="",CUDA_HOME=str(cuda),HUMMING_DEBUG_KERNEL="0",
                  PATH=str(Path(sys.prefix)/"bin")+":"+str(cuda/"bin")+":"+os.environ["PATH"])
sys.path.insert(0,str(root/"swarm_2026-11-16/moe/humming_vendor"))
import torch
from humming.jit.compiler import NVRTCCompiler
from scale_order import adapted_source
assert not torch.cuda.is_initialized()
old=root/"swarm_2026-11-17/deep_research/hopper_moe"
records=[]
for stage,folder in (("w13",old/"offline_occupancy_cpasync/w13_cp_async_m8_s3_cta1"),
                     ("w2",old/"offline_k_preserving/w2_m64_k64_wk64")):
    source=(folder/"kernel.cu").read_text()
    assert "#if 0\n#include <humming/kernel/humming_ws.cuh>" in source and "kUseStreamK = false" in source
    code=source.replace("#include <humming/kernel/humming.cuh>",adapted_source())
    file=Path(NVRTCCompiler.compile(code,sm_version="90a",kernel_expr=(folder/"kernel_expression.txt").read_text().strip(),disable_fast_math=False))
    local=here/(stage+"_sm90.cubin");shutil.copyfile(file,local)
    records.append(dict(stage=stage,cubin_sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
                        bytes=local.stat().st_size,source=str(folder.relative_to(root))))
assert not torch.cuda.is_initialized()
(here/"offline.json").write_text(json.dumps(records,indent=2)+"\n");print(records)

