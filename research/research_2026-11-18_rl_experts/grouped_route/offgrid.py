"""Recompute predeclared off-grid decisions on pinned host-side tuning heuristics."""
import importlib.util
import json
import os
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0,str(root/"swarm_2026-11-16/moe/humming_vendor"))
import torch
from humming.config import LayerConfig,GemmType
from humming.tune.sm8x import Sm86Heuristics
from route_regime import route_regime, dense_route_tiles
from unittest.mock import patch
archived=json.loads((root/"swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json").read_text())["stages"]
threshold={r["kwargs"]["shape_n"]:r["cached_result"] for r in json.loads((here/"fixture_threshold_cache.json").read_text())["results"]}
gates=json.loads((here/"package.json").read_text())["geometries"]
def for_arch(sm,m):
    light={};heavy={}
    for stage,record in archived.items():
        def cfg(size):
            if sm==90:
                import copy
                return copy.deepcopy(next(c for lo,hi,c in record["intervals"] if lo<size<=hi))
            meta=LayerConfig(**dict(record["meta"],mma_type="mma"))
            with patch.object(LayerConfig,"estimate_bound_min_shape_m",lambda obj,flag:threshold[obj.shape_n]),patch.object(Sm86Heuristics,"get_num_sms",return_value=20):
                return Sm86Heuristics.get_config(meta,size,use_f16_accum=False,gemm_type=GemmType.INDEXED)
        a,b=cfg(m),cfg(m*8)
        for c in (a,b):c.update(use_stream_k=False,use_f16_accum=False)
        if m*8>4096:
            for key in ("block_shape","warp_shape"):
                a[key]=list(a[key]);a[key][2]=b[key][2]
        light[stage]=a;heavy[stage]=b
    if light==heavy:return dict(mode="indexed_large" if dense_route_tiles(m*8,256,heavy) else "indexed",reason="default layout has no distinct light partition")
    cutoff=min(c["block_shape"][0] for c in light.values())
    mode=route_regime(m*8,32,256,dict(light=light,heavy=heavy,cutoff=cutoff))
    raw=mode
    keys=[repr(((8,6) if sm==86 else (9,0),stage,tuple(c["block_shape"]),tuple(c["warp_shape"]),c["num_stages"])) for p in (light,heavy) for stage,c in p.items()]
    if mode=="grouped" and any(k not in gates for k in keys):mode="indexed"
    return dict(mode=mode,raw=raw,cutoff=cutoff,heavy_m=max(c["block_shape"][0] for c in heavy.values()))
sizes=[31,63,119,127,128,129,143,255,511,513,735,767,768,769,800,1025,3013,4095,4096,4097,4141,8189,8193,16383,16384]
rows={str(arch):{str(m):for_arch(arch,m) for m in sizes} for arch in (86,90)}
local=rows["86"]
assert local["128"]["mode"]=="grouped" and local["768"]["mode"]=="grouped"
assert local["31"]["mode"]!="grouped" and local["16384"]["mode"]=="indexed_large"
assert rows["90"]["16384"]["mode"]=="indexed_large"
assert rows["90"]["128"]["mode"]=="grouped" and rows["90"]["768"]["mode"]=="indexed"
assert not torch.cuda.is_initialized()
(here/"offgrid_policy.json").write_text(json.dumps(dict(scope="CPU-only eligibility from the archived pinned H200 heuristic table and local SM86 source with measured prepare-time thresholds; no GPU timing",rows=rows),indent=2))
print({a:{m:v["mode"] for m,v in d.items()} for a,d in rows.items()})
