"""Select one spare whole-expert copy per rank in past data, then route using current counts."""
import json
from pathlib import Path
import numpy as np
from placement import NATIVE, work, route_counts
from real_traces import LAYERS,STARTS

HERE=Path(__file__).resolve().parent
RANKS=8
TILE=16


def transfers(counts,edges):
    # Transfer complete tiles; a residual non-multiple stays at its physical home.
    native=work(counts,TILE)
    load=native.reshape(RANKS,-1).sum(axis=1)
    assigned=np.zeros(len(edges),dtype=int)
    for _ in range(8):
        for k,(e,dst) in enumerate(edges):
            src=e//32
            capacity=counts[e]//TILE
            delta=int((load[src]-load[dst])/2)
            delta=max(-assigned[k],min(capacity-assigned[k],delta))
            load[src]-=delta
            load[dst]+=delta
            assigned[k]+=delta
    assert np.all(assigned>=0) and np.all(load>=0) and load.sum()==native.sum()
    return int(load.max()),assigned


def plan(counts,budget):
    # Favor expensive experts on stressed ranks, choosing remote slots with most headroom.
    loads=work(counts,TILE).reshape(RANKS,-1).sum(axis=1)
    edges=[]
    free=set(range(RANKS))
    placed=set()
    while free and len(edges)<budget:
        best=None
        for e in np.argsort(counts)[-32:]:
            if int(e) in placed:continue
            src=int(e)//32
            for dst in sorted(free-{src}):
                moved=min(counts[e]//TILE,(loads[src]-loads[dst])//2)
                gain=moved*(loads[src]-loads[dst]-moved)
                if moved>0 and (best is None or gain>best[0]):
                    best=(gain,int(e),int(dst),moved)
        if best is None:break
        _,e,dst,moved=best
        loads[e//32]-=moved
        loads[dst]+=moved
        free.remove(dst)
        placed.add(e)
        edges.append((e,dst))
    return edges


def main():
    all_results={}
    for name in ("glm53","glm52"):
        arr=np.load(HERE/(name+"_routes.npz"))["arr"]
        summary={}
        for m in (128,768):
            for budget in (2,4,8):
                data=[]
                for region in range(len(STARTS)):
                    for l in range(len(LAYERS)):
                        ids=arr[region,:,l]
                        edges=plan(route_counts(ids[:2048]),budget)
                        for off in range(2048,len(ids)-m+1,m):
                            counts=route_counts(ids[off:off+m])
                            native,unused=transfers(counts,[])
                            improved,moved=transfers(counts,edges)
                            data.append((native,improved,int(sum(moved))*TILE))
                before,after,changed=np.array(data).T
                assert np.all(after<=before)
                summary[f"M{m}_copies{budget}"]=dict(cases=len(before),peak_tile_ratio=float(before.sum()/after.sum()),
                    fraction_helped=float((after<before).mean()),fraction_routes_transferred=float(changed.sum()/(len(before)*m*8)))
        all_results[name]=summary
    (HERE/"replica_screen.json").write_text(json.dumps({"scope":"No migration or mapping costs. One previously chosen spare expert per eligible receiver. Not a time or H200 estimate.","results":all_results},indent=2))
    print(json.dumps(all_results))

if __name__=="__main__":main()
