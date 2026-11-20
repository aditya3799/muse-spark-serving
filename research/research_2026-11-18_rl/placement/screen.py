"""Measure projected *rank work*, migration size and stability for held-out route inputs."""
import json
from pathlib import Path
import numpy as np
from placement import NATIVE, EXPERT_BYTES, demand, draw, peak, select, route_counts, local_map

HERE=Path(__file__).resolve().parent


def evaluate(seed, pattern, rows, swap_limit):
    heat=demand(seed,pattern)
    training=draw(seed+99,2048,heat)
    slots,swaps=select(route_counts(training),swap_limit)
    moved=[np.flatnonzero(a!=b).size for a,b in zip(slots,NATIVE)]
    record=dict(swaps=len(swaps),peak_moved_bytes=round(max(moved)*EXPERT_BYTES),tests={})
    for case,latent in [("same",heat),("half",.5*heat+.5*demand(seed+1,pattern)),("new",demand(seed+1,pattern))]:
        probe=draw(seed+rows+3,rows,latent)
        counts=route_counts(probe)
        for i in range(8):
            local=local_map(slots,i)[probe]
            mask=local>=0
            assert np.array_equal(probe[mask], slots[i,local[mask]])
        record["tests"][case]={str(tile):peak(counts,NATIVE,tile)/peak(counts,slots,tile) for tile in (1,16,64)}
    return record


def main():
    summary={}
    for pattern in ("normal","skewed","uniform"):
        for m in (128,768,3072):
            for limit in (1,4,8,16):
                cases=[evaluate(seed,pattern,m,limit) for seed in range(1201,1213)]
                reduced={}
                for agreement in ("same","half","new"):
                    values=[r["tests"][agreement]["16"] for r in cases]
                    reduced[agreement]={"median":float(np.median(values)),"min":min(values),"max":max(values)}
                summary[f"{pattern}_M{m}_swaps{limit}"]=dict(peak_rank_tile16_ratios=reduced,
                    max_moved_bytes_per_rank=max(r["peak_moved_bytes"] for r in cases),cases=cases)
    result=dict(scope="Synthetic public routing laws reimplemented in NumPy. Held-out router scores, max rank work proxy only. No serving timing, training cost or NCCL measured.",source="muse_moe_w4a16/reference.py",results=summary)
    (HERE/"screen.json").write_text(json.dumps(result,indent=2))
    for pattern in ("normal","skewed","uniform"):
        print(pattern,[(m,summary[f"{pattern}_M{m}_swaps4"]["peak_rank_tile16_ratios"]["same"]["median"],summary[f"{pattern}_M{m}_swaps4"]["peak_rank_tile16_ratios"]["new"]["median"]) for m in (128,768,3072)])


if __name__=="__main__":main()
