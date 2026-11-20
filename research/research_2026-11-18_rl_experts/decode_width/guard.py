"""Count which retained routes have unknowable original Humming tile ownership."""
from pathlib import Path
import json
import hashlib
import numpy as np

here=Path(__file__).resolve().parent
root=here.parents[1]
archives={86:root/"swarm_2026-11-16/projection_communication/phase4_hour/profile_run_01/owned_configs.json",90:root/"swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json"}
actual=json.loads(archives[86].read_text())["indexed"]
modeled={k:v["intervals"] for k,v in json.loads(archives[90].read_text())["stages"].items()}
held=json.loads((root/"research_2026-11-18_rl/lattice_tree/gpu_widths_holdout.json").read_text())
def geometry(table,m):
    plan={}
    for stage,intervals in table.items():
        def lookup(routes):
            return next(c for lo,hi,c in intervals if lo<routes<=hi)
        light=lookup(m)
        heavy=lookup(8*m)
        plan[stage]=dict(light=[light["block_shape"],light["warp_shape"]],heavy=[heavy["block_shape"],heavy["warp_shape"]])
    return plan
def kpair(plan,stage,name):
    block,warp=plan[stage][name]
    return block[2],warp[2]
result=dict(scope="Seeded CPU route fixtures use the same number of global experts, top-k and even-row skew as the GPU harness, with independent RNG. Width masks are the held-out realized masks; these are not measured serving routes. SM90 table is the pinned architecture-mocked heuristic and does not establish actual load-time dispatch.",cases={},inputs={str(k):hashlib.sha256(v.read_bytes()).hexdigest() for k,v in archives.items()})
for sm,table in [(86,actual),(90,modeled)]:
    for b in (16,32):
        full=geometry(table,8*b)
        cut=min(full[s]["light"][0][0] for s in ("w13","w2"))
        k_same=all(kpair(full,s,"light")==kpair(full,s,"heavy") for s in ("w13","w2"))
        for pattern in ("uniform","skewed"):
            source=held["shapes"][f"B{b}_{pattern}"]["rows"]
            counts=dict(batches=0,uncertified_batches=0,uncertified_rows=0,retained=0,uncertified_routes=0,retained_local_routes=0,actually_misclassified_routes=0)
            for seed in (309611,827449):
                rng=np.random.default_rng(seed)
                for r in source:
                    dense=8*b
                    mask=np.arange(dense)%8<np.repeat(r["widths"],8)
                    keys=rng.random((dense,256),dtype=np.float32)
                    if pattern=="skewed":keys[::2,:8]+=1.
                    ids=np.argpartition(keys,-8,axis=-1)[:,-8:]
                    valid=ids[mask]<32;remaining=dense-int(mask.sum())
                    subset=ids[mask]
                    observed=np.bincount(subset[valid],minlength=256)
                    total=np.bincount(ids[ids<32],minlength=256)
                    ambiguous=(observed<=cut)&(observed+remaining>cut)
                    disagree=(observed<=cut)&(total>cut)
                    flagged=valid&ambiguous[subset]&(not k_same)
                    counts["batches"]+=1
                    counts["uncertified_batches"]+=int(flagged.any())
                    counts["uncertified_rows"]+=int(flagged.any(1).sum())
                    counts["retained"]+=int(mask.sum())
                    counts["uncertified_routes"]+=int(flagged.sum())
                    counts["retained_local_routes"]+=int(valid.sum())
                    counts["actually_misclassified_routes"]+=int((valid&disagree[subset]&(not k_same)).sum())
            result["cases"][f"SM{sm}/B{b}/{pattern}"]=dict(frozen=full,compact=geometry(table,4*b),same_reduction_topology_across_dense_partition=k_same,cutoff=cut,**counts)
(here/"guard.json").write_text(json.dumps(result,indent=2))
for k,r in result["cases"].items():print(k,"uncertified_batches",r["uncertified_batches"],"rows",r["uncertified_rows"],"/",r["retained"],"actual lost labels",r["actually_misclassified_routes"],flush=True)
