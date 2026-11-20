"""Compare the submitted Hopper token chunks with global 64-route grouping.

The counters below measure jobs, requested global-to-shared weight bytes and the
packaged scheduler's relative work model. They do not measure HBM or wall time.
"""
import ast
import hashlib
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = ROOT / "research_2026-11-18_target_moe/native_package_02"
NATIVE = (PACKAGE / "hopper_expert.cu").read_text(encoding="ascii")
PATCH = "1ce48266f47574c5c0d4fd3c557b12f6d24811d03121f9bd9ea4e89965f6065c"
assert hashlib.sha256((PACKAGE/"candidate.diff").read_bytes()).hexdigest()==PATCH
assert "constexpr int FT_T = 64;" in NATIVE and "constexpr int MMAX = 384" in NATIVE
assert "for first in range(0,rows,384)" in (PACKAGE/"hopper_backend.py").read_text()
def costs(stage):
    return list(map(int,re.search("WGS_C"+stage+r"\[8\] = \{([^}]+)",NATIVE).group(1).split(",")))
WA,WB = costs("A"), costs("B")
EXPERT_BYTES = (8192+4096)*(2048+256)

def summarize(jobs):
    waste=sum((-len(slots))%8 for _,slots in jobs)
    cost=sum(2*WA[(len(s)-1)//8]+WB[(len(s)-1)//8] for e,s in jobs)
    return dict(jobs=len(jobs),route_slots=sum(len(s) for _,s in jobs),padded_rows=waste,
                global_to_shared_weight_bytes=len(jobs)*EXPERT_BYTES,
                scheduler_cost=cost)

def plan(ids, chunk):
    jobs=[]
    for begin in range(0,len(ids),chunk):
        slots=[[] for _ in range(32)]
        for row,topk in enumerate(ids[begin:begin+chunk],begin):
            for lane,e in enumerate(topk):
                if e>=0: slots[e].append(row*8+lane)
        for e,routes in enumerate(slots):
            jobs.extend((e,routes[i:i+64]) for i in range(0,len(routes),64))
    return jobs

def check(ids,jobs):
    live={8*row+slot:e for row,top in enumerate(ids) for slot,e in enumerate(top) if e>=0}
    restored={}
    for e,slots in jobs:
        assert 1<=len(slots)<=64
        for slot in slots:
            assert slot not in restored
            restored[slot]=e
    assert restored==live
    # A 48-group window uses at most 3072 row slots, the existing scratch capacity.
    assert all(sum((len(slots)+7)//8*8 for _,slots in jobs[i:i+48])<=3072 for i in range(0,len(jobs),48))
    # Distinct dyadic values in route lanes detect accidental slot permutations.
    assert sum((slot+1)*(e+3)*(slot%8+1) for slot,e in restored.items()) == sum((slot+1)*(e+3)*(slot%8+1) for slot,e in live.items())

def scenarios():
    src=ROOT.parent/"muse_prefill_layer/submission_n72_exactroute_fullkperm_candidate.py"
    definition=next(x.value for x in ast.parse(src.read_text()).body if isinstance(x,ast.Assign) and any(getattr(t,"id","")=="SCEN" for t in x.targets))
    for _,(m,counts) in ast.literal_eval(definition).items():
        ids=[[-1]*8 for _ in range(m)]
        offset=0
        for e,n in enumerate(counts):
            for i in range(offset,offset+n): ids[i%m][i//m]=e
            offset+=n
        yield "kernel_track_donor_fixture",m,0,ids
        for seed in (12,25,72,103):
            rng=random.Random(seed)
            ids_sh=ids[:]
            rng.shuffle(ids_sh)
            yield "donor_histogram_shuffled",m,seed,ids_sh
            uniform=[tuple(v if v<32 else -1 for v in rng.sample(range(256),8)) for _ in range(m)]
            yield "uniform_global_top8",m,seed,uniform
            dense=[rng.sample(range(32),8) for _ in range(m)]
            yield "all_local",m,seed,dense
    yield "all_remote",385,0,[[-1]*8 for _ in range(385)]

def main():
    rows=[]
    for pattern,m,seed,ids in scenarios():
        a,b=plan(ids,384),plan(ids,m)
        check(ids,a)
        check(ids,b)
        before,after=summarize(a),summarize(b)
        rows.append(dict(pattern=pattern,rows=m,seed=seed,before=before,after=after,
                         old_token_chunks=(m+383)//384,new_nonempty_tile_windows=(len(b)+47)//48,
                         byte_ratio=(len(a)/len(b)) if b else None,
                         scheduler_proxy_ratio=before["scheduler_cost"]/after["scheduler_cost"] if b else None))
    run=json.loads((ROOT/"release_2026-11-18/scored_submission_status.json").read_text())
    shp=run["result"]["shape"]
    n,first,new,answer=shp["turns"],shp["first_turn"],shp["new_per_turn"],shp["output_mean"]
    unique=first+(n-1)*new
    presented=sum(first+i*(new+answer) for i in range(n))
    data=dict(patch=PATCH,scored_parent=run["id"],route_scenarios=rows,
              cold_prompt_share=first/unique,
              full_retention_hit_rate_proxy=1-unique/presented,
              parent_actual_hit_rates=[r["cache_hit"] for r in run["result"]["rungs"]],
              interpretation="The full-retention workload model uses mean lengths and counts all prompts including history. Job traffic counts requested weight bytes before cache. Scheduler weights estimate relative packing costs inherited from the source, not exact package timing.")
    (HERE/"route_bounds.json").write_text(json.dumps(data,indent=2)+"\n")
    for m in (753,3012,16384):
        for row in rows:
            if row["rows"]==m and row["seed"] in (0,12) and row["pattern"]!="donor_histogram_shuffled":
                print(m,row["pattern"],row["before"]["jobs"],row["after"]["jobs"],round(row["byte_ratio"],3),round(row["scheduler_proxy_ratio"],3))

if __name__=="__main__":
    main()
