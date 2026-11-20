"""Admit only resources qualified for full-K 512-column vendor kernels."""
import json
from functools import lru_cache
from pathlib import Path
@lru_cache()
def allowed():
    rows=json.loads((Path(__file__).with_name("resources_checked.json")).read_text())
    return {(r["sm"],r["stage"],tuple(r["block"]),tuple(r["warp"]),r["stages"]) for r in rows if r["allowed"]}
def wide(config,sm,valid,stage,variant):
    if variant not in (stage,"both") or sm not in (86,90):return config
    if config.get("use_f16_accum") is not False or config.get("use_stream_k") is not False:return config
    if config.get("use_tma") or config.get("use_warp_spec") or config.get("num_write_splits",1)!=1:return config
    if any(config.get(k,1)!=1 for k in ("multi_cast_size_a","multi_cast_size_b","num_ctas_per_sm")):return config
    bm,bn,bk=config["block_shape"];warp=tuple(config["warp_shape"])
    if bn!=256:return config
    key=(sm,stage,(bm,512,bk),warp,config.get("num_stages",2))
    return dict(config,block_shape=[bm,512,bk]) if key in allowed() else config


