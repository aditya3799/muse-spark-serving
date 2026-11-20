"""Model every row/feature transfer, counter epoch, and BF16 wire value on CPU."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import numpy as np
import build

here=Path(__file__).resolve().parent
world=8
hidden=8192
local=hidden//world
slots=132
row_inputs=[8,32,128,256,1,9,192,511,7,64]
half=np.zeros((2,world,511,local),dtype=np.uint16)
counters=np.zeros(slots,dtype=np.int32)
rng=np.random.default_rng(985367)
report={"seed":985367,"cases":[]}
for rows in row_inputs:
    original=rng.integers(0,65536,size=(world,rows,local),dtype=np.uint16)
    original[:,:,::61]=0
    full=np.transpose(original,(1,0,2)).reshape(rows,hidden)
    wire=np.where(original==0,0x8000,original).astype(np.uint16)
    expected=np.where(full==0,0x8000,full).astype(np.uint16)
    phase=counters[0]&1
    assert (counters == phase).all()
    grid=min(rows,slots);threads=256 if rows<192 else 512
    arena=half[phase,:,:rows,:]
    assert not arena.any()
    # Simulate the actual global-stride push with randomized CTA ordering.
    for block in rng.permutation(grid):
        for first in range(block*threads*8,rows*local,grid*threads*8):
            end=min(first+threads*8,rows*local)
            rr,cc=np.divmod(np.arange(first,end),local)
            arena[:,rr,cc]=wire[:,rr,cc]
    out=np.empty((rows,hidden),dtype=np.uint16)
    for block in rng.permutation(grid):
        for r in range(block,rows,grid):
            # receive_pack's two-vector prefetch and staggered zero reset stay in this row.
            for pp in range(hidden//(threads*8)):
                feats=pp*threads*8+np.arange(threads*8)
                owners,within=np.divmod(feats,local)
                vals=arena[owners,r,within]
                assert np.all(vals!=0)
                out[r,feats]=vals
                arena[owners,r,within]=0
        counters[block]=phase^1
    counters[grid:]=phase^1
    assert np.array_equal(out,expected) and not arena.any()
    report["cases"].append(dict(rows=rows,threads=threads,grid=grid,phase=int(phase),full_bf16_words=int(out.size),mismatches=int(np.count_nonzero(out!=expected))))
with tempfile.TemporaryDirectory() as tmp:
    td=Path(tmp)
    for p in [build.model,build.gather]:
        f=td/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(build.get(p).encode())
    subprocess.run(["git","apply","--check",str(here/"incremental.diff")],cwd=td,check=True)
report["patch_applies_to_original_sources"]=True
report["scope"]="All source communication addresses, word encoding, per-block known phase and reset. Simulation separates producer/consumer stages; no cross-GPU progress or operation timing is inferred."
report["source_provenance"]={}
for p in ["csrc/muse_spark/all_gather_norm_residual.cuh","csrc/muse_spark/norm_residual.cuh"]:
    rel="python/sglang/kernels/jit/"+p
    frozen=build.get(rel).replace("\r\n","\n")
    compiled=(build.repo/rel).read_text().replace("\r\n","\n")
    assert frozen==compiled
    report["source_provenance"][rel]=hashlib.sha256(frozen.encode()).hexdigest()
(here/"protocol.json").write_text(json.dumps(report,indent=2))
print("word comparisons",sum(r["full_bf16_words"] for r in report["cases"]),"all passed")
