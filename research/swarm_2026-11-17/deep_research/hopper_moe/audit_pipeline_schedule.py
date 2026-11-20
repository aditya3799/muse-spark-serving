"""Source-level schedule invariant for non-WS pipeline depths 3 and 4."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent

def schedule(stages,k_blocks,k_partitions,warp_iters):
    # Each entry tracks the logical K-block resident at that stage slot.
    slots=[None]*stages
    next_load=0
    loads=[]
    def load(slot,predicate):
        nonlocal next_load
        if predicate:
            slots[slot%stages]=next_load
            loads.append(next_load)
        next_load+=1
    load(0,True)
    for s in range(1,max(stages-1,2)):
        load(s,s<k_blocks)
    sequence=[]
    remaining=k_blocks
    while remaining:
        for s in range(stages):
            logical=k_blocks-remaining
            assert slots[s]==logical,(stages,k_blocks,logical,slots)
            for it in range(warp_iters):
                sequence.extend((p,logical*k_partitions*warp_iters+p*warp_iters+it)
                                for p in range(k_partitions))
                if it==warp_iters-2:
                    load(s+stages-1,remaining>=stages)
            remaining-=1
            if not remaining:break
    assert loads==list(range(k_blocks))
    return [[k for p2,k in sequence if p2==p] for p in range(k_partitions)]

checks=[]
for kb in (1,2,3,4,5,8,16,31,32,33,64,65):
    for partitions in (1,2,4):
        for warp_iters in (2,4,8):
            a=schedule(3,kb,partitions,warp_iters)
            b=schedule(4,kb,partitions,warp_iters)
            assert a==b
            checks.append({'K_blocks':kb,'K_partitions':partitions,'warp_iters':warp_iters})
r={'status':'PASS_ABSTRACT_NON_WS_K_SEQUENCE','checks':len(checks),
   'scope':'Symbolic pinned producer/consumer traversal, not floating-point or hardware proof',
   'invariant':'Depth3 and depth4 execute the same K-slab sequence per partition with identical padding and FP32 reductions when blockK/warpK remain fixed.',
   'remaining':'Native compiler outputs and actual H200 numerical/graph behavior still require execution.',
   'cases':checks}
(P/'pipeline_schedule_results.json').write_text(json.dumps(r,indent=2)+'\n')
print(r['status'],r['checks'])
