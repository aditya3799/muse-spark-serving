"""Conditional fixed-BM planner ablation; no runtime/latency prediction."""
from replica_ownership_oracle import Session,Budget,analyze
import pathlib,json,random,hashlib
P=pathlib.Path(__file__).parent

def choose_tiles(session,budget,max_copies=8,bm=16):
    hs=[list(h)for h in session.baseline().holders]
    def score(hs):
        routes=[0]*session.ranks;tiles=[0]*session.ranks
        for e,owners in enumerate(hs):
            q,rem=divmod(session.counts[e],len(owners))
            for j,r in enumerate(owners):
                n=q+(j<rem);routes[r]+=n;tiles[r]+=(n+bm-1)//bm
        return(max(tiles),sum(tiles),max(routes),sum(v*v for v in routes))
    for step in range(max_copies):
        old=score(hs);best=None
        for e in range(session.experts):
            if not session.counts[e]:continue
            for r in range(session.ranks):
                if r in hs[e] or budget.live[r]+2*28_311_552>budget.limit:continue
                trial=[h[:]for h in hs];trial[e].append(r);trial[e].sort();sc=score(trial)
                if sc<old and (best is None or (sc,e,r)<best[:3]):best=(sc,e,r,trial)
        if best is None:break
        _,e,r,hs=best;budget.reserve(('tile',session.run_id,session.epoch,e,r),r,28_311_552,28_311_552)
    return session._plan(hs)

def main():
    rng=random.Random(78941);cases=[]
    patterns={
      'eight_hot':[list(range(8))for _ in range(128)],
      'balanced':[[(t*8+k)%256 for k in range(8)]for t in range(128)],
      'random':[rng.sample(range(256),8)for _ in range(128)],
      'strided':[[0]+rng.sample(range(32,256),7) if t%8==0 else rng.sample(range(32,256),8)for t in range(128)],
    }
    for name,train in patterns.items():
        ses=Session();ses.observe(train);bp=ses.baseline()
        planners={'count':ses.choose(Budget(8),8),'tile':choose_tiles(ses,Budget(8))}
        for evaluate,rows in [('observed',train),('shifted_random',[rng.sample(range(256),8)for _ in range(128)])]:
            baseline=analyze(rows,bp,ses.route(rows,bp))
            for planner,plan in planners.items():
                for route in ['ordinal','token_mod']:
                    result=analyze(rows,plan,ses.route(rows,plan,route))
                    cases.append(dict(train=name,eval=evaluate,planner=planner,ownership=route,copies=sum(len(h)-1 for h in plan.holders),baseline=baseline,candidate=result))
    # Adversarial token positions: all occurrences of expert0 are even. A two-copy
    # token-mod map sends all work to its first holder, unlike occurrence ordinal.
    ses=Session();rs=[[0]+list(range(32,39))if t%2==0 else list(range(32,40))for t in range(32)];ses.observe(rs)
    h=[list(v)for v in ses.baseline().holders];h[0]=[0,2];pl=ses._plan(h)
    ordinal=ses.route(rs,pl,'ordinal');mod=ses.route(rs,pl,'token_mod')
    def e0load(ids):return [sum(ids[r][t][0]>=0 for t in range(0,32,2))for r in range(8)]
    assert e0load(ordinal)==[8,0,8,0,0,0,0,0] and e0load(mod)==[16,0,0,0,0,0,0,0]
    report={'status':'CPU_TILE_OBJECTIVE_FALSIFICATION','scope':'FixedBM16 structural proxy corresponding to prior local SM86 M32 light/heavy BM; not full native config or runtime. All counts actual legal routes in synthetic same-run fixtures.','cases':cases,'token_mod_counterexample':{'ordinal_expert0_load':e0load(ordinal),'token_mod_expert0_load':e0load(mod)},'interpretation':['Count-only placement can add padded tiles with no max-tile benefit. Tile-aware admission is a stronger necessary screen, not sufficient performance evidence.','Training-count balanced split estimates may differ from token-mod actual assignment. Current-batch ownership must be evaluated or use a better general map; stale-count distribution changes remain a risk.','Real full-helper timing includes histogram/mapping/copy amortization, unchanged EP collective, kernel wave scheduling and cache effects.']}
    (P/'TILE_PLACEMENT_ABLATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(cases),'token_mod_counterexample':report['token_mod_counterexample']}))
if __name__=='__main__':main()
