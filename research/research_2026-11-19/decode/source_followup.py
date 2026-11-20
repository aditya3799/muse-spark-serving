"""Read selected upstream files, with explicit per-file and total byte bounds."""
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent/"sources"
repo=HERE/"deepinfer_git"
paths=[
 "deepinfer/model_executor/input_metadata/dflash.py",
 "deepinfer/model_executor/models/mtp/dflash.py",
 "deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py",
 "deepinfer/model_executor/layers/tilelang_ops/cuda/dflash.py",
 "deepinfer/model_executor/layers/tilelang_ops/cuda/dflash_copy.py",
 "deepinfer/worker/dflash_cuda_graph_runner.py",
 "deepinfer/worker/decode_perf_cache.py",
 "deepinfer/worker/model_runner.py",
 "deepinfer/model_executor/layers/linear.py",
 "deepinfer/model_executor/input_metadata/yoco.py",
 "csrc/engine/mtp_varlen.hpp",
 "csrc/engine/process_output.hpp",
 "tests/models/test_batch_invariant.py",
]
subprocess.run(["git","sparse-checkout","set","--no-cone","--stdin"],cwd=repo,
 input="\n".join("/"+p for p in paths)+"\n",text=True,check=True,timeout=90)
subprocess.run(["git","checkout","HEAD"],cwd=repo,check=True,timeout=90)
commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
manifest=[]
for p in paths:
 data=(repo/p).read_bytes()
 manifest.append(dict(repo="deepseek-ai/DeepInfer",commit=commit,path=p,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
assert sum(x["bytes"] for x in manifest)<3_000_000
flash="f89bc2306632d1ec5f97b014dded4254f5b4a907"
for p in ["flash_api.cpp","seqlen.h","mask.h","block.h","flash_fwd_kernel_sm80.h","mainloop_fwd_sm80.hpp","tile_size.h"]:
 url=f"https://raw.githubusercontent.com/sgl-project/sgl-flash-attn/{flash}/hopper/{p}"
 try:
  with urllib.request.urlopen(url,timeout=30) as response:data=response.read(300_001)
  assert len(data)<=300_000
  target=HERE/"sgl-flash-attn"/p;target.parent.mkdir(exist_ok=True);target.write_bytes(data)
  manifest.append(dict(repo="sgl-project/sgl-flash-attn",commit=flash,path="hopper/"+p,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 except Exception as error:manifest.append(dict(url=url,error=str(error)))
(HERE/"followup_manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
