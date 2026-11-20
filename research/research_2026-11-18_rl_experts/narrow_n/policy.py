"""Guarded isolated CTA output-column tuning; the vendor full-K kernel does all math."""
def narrow(config,sm,valid,stage,variant):
    if variant not in (stage,"both") or sm not in (86,90):
        return config
    if config.get("use_f16_accum") is not False or config.get("use_stream_k") is not False:
        return config
    if config.get("use_tma") or config.get("use_warp_spec") or config.get("num_write_splits",1)!=1:
        return config
    if any(config.get(k,1)!=1 for k in ("multi_cast_size_a","multi_cast_size_b")):
        return config
    bm,bn,bk=config["block_shape"];wm,wn,wk=config["warp_shape"]
    if bn!=256 or 128%wn or (sm==90 and (128//wn)%4):
        return config
    # Keep kernel vector-copy cohorts whole and avoid sub-warp or single-warp launches.
    threads=(bm//wm)*(128//wn)*(bk//wk)*32
    if not 64<=threads<=1024:
        return config
    return dict(config,block_shape=[bm,128,bk])

