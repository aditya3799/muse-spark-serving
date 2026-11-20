import json,pathlib,random,hashlib,time
from fractions import Fraction
from replica_ownership_oracle import Session,Budget,analyze
P=pathlib.Path(__file__).parent
rng=random.Random(9018)
def rows(kind,m):
    out=[]
    for t in range(m):
        if kind=='balanced':r=[(t*8+k)%256 for k in range(8)]
        elif kind=='eight_hot':r=list(range(8))
        elif kind=='one_hot':r=[0]+[32*k+(t%32) for k in range(1,8)]
        elif kind=='strided':r=([0]+rng.sample(range(32,256),7)) if t%8==0 else rng.sample(range(32,256),8)
        elif kind=='random':r=rng.sample(range(256),8)
        elif kind=='invalid':r=[-1]*8
        out.append(r)
    return out
report={'scope':'CPU routing/placement oracle; costs are structural proxies, not measured runtime','cases':[],'negatives':[]}
plans=[]
for kind in ['balanced','eight_hot','one_hot','strided','random']:
    s=Session();train=rows(kind,128);s.observe(train);budget=Budget(8)
    plan=s.choose(budget,max_copies=8,layer=kind);plans.append((s,plan))
    for eval_kind in ['observed','same_distribution','balanced','invalid']:
        rs=train if eval_kind=='observed' else rows(kind if eval_kind=='same_distribution' else eval_kind,128);base=analyze(rs,s.baseline(),s.route(rs,s.baseline()))
        for policy in ['ordinal','token_mod','whole']:
            ids=s.route(rs,plan,policy);metrics=analyze(rs,plan,ids)
            # Every local slot refers to a byte-identical small packed fixture.
            source={e:bytes((e*13+i*7)%256 for i in range(37)) for e in range(256)}
            storage=[{plan.slots[r][e]:source[e] for e,hs in enumerate(plan.holders) if r in hs}for r in range(8)]
            for t,row in enumerate(rs):
                expected=sum((Fraction((e+1)*(t+1),k+1) for k,e in enumerate(row) if e>=0),Fraction())
                actual=Fraction()
                for r in range(8):
                    for k,e in enumerate(row):
                        slot=ids[r][t][k]
                        if slot>=0:
                            assert storage[r][slot]==source[e]
                            actual+=Fraction((e+1)*(t+1),k+1)
                assert actual==expected
            report['cases'].append(dict(train=kind,eval=eval_kind,policy=policy,baseline=base,candidate=metrics,copies=sum(len(h)-1 for h in plan.holders),peak_bytes_per_gpu=budget.peak))
# Random changed batches preserve ownership even after distribution shifts.
for i in range(100):
    s,plan=plans[i%len(plans)];rs=[rng.sample(range(256),8) for _ in range(rng.randrange(1,40))]
    if i%3==0:rs[-1][-2:]=[-1,-1]
    analyze(rs,plan,s.route(rs,plan,'token_mod'))
report['changed_random_batches']=100
for name,fn in [('cross_run',lambda:Session().route([[0]],plans[0][1])),('duplicate_route',lambda:plans[0][0].observe([[1,1]])),('bad_id',lambda:plans[0][0].observe([[256]])),('peak_budget',lambda:Budget(1).reserve('x',0,1_999_999_999,2))]:
    try:fn()
    except ValueError:report['negatives'].append(name)
    else:raise AssertionError(name)
# Multiple layers share one per-GPU budget, and transfer temporary counts.
b=Budget(1,100);b.reserve('layerA',0,60,20)
try:b.reserve('layerB',0,30,20)
except ValueError:report['negatives'].append('cross_layer_transfer_peak')
else:raise AssertionError('cross layer')
b.release('layerA');b.reserve('layerB',0,30,20);assert b.peak==[80]
# Deliberately executing every resident replica violates exactly once.
s,pl=plans[1];rs=[[0]];ids=[[[-1]]for _ in range(8)]
for r in pl.holders[0]:ids[r][0][0]=pl.slots[r][0]
try:analyze(rs,pl,ids)
except AssertionError:report['negatives'].append('naive_all_copies_double_count')
else:raise AssertionError('negative not exercised')
report['status']='PASS_CPU_OWNERSHIP_AND_LOSSLESS_FIXTURE_ONLY'
report['source_hashes']={n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in ['replica_ownership_oracle.py','test_replica_oracle.py']}
(P/'REPLICA_ORACLE_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'cases':len(report['cases']),'random_batches':100,'negatives':report['negatives']}))
