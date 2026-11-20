"""Offline speculation economics. No torch, model imports, GPU or policy integration.

Optional measurement JSON: {"semantics":"unchanged_draft_prefix", "histogram":[n1,...,n8],
"cycle_us_by_verify_width":{"4":...,"6":...,"8":...}}.
Costs must include the same full draft pass, target verification and commit work.
A histogram from block8 does NOT predict changed drafter-block4 acceptance.
"""
import argparse,json

def bounds(mean,width,original=8):
    assert 1<=mean<=original and 1<=width<=original
    # For L in [1,N], min(L,k) lies above the endpoint chord and below min(E[L],k).
    return {'min':1+(mean-1)*(width-1)/(original-1),'max':min(mean,width)}

def analyze(measurement):
    assert measurement['semantics']=='unchanged_draft_prefix'
    hist=measurement['histogram'];assert len(hist)==8 and all(isinstance(x,int) and x>=0 for x in hist) and sum(hist)>0
    cost={int(k):float(v) for k,v in measurement['cycle_us_by_verify_width'].items()}
    assert 8 in cost and all(v>0 for v in cost.values())
    mean=sum((i+1)*n for i,n in enumerate(hist))/sum(hist)
    rows=[]
    for width in sorted(cost):
        assert 1<=width<=8
        expected=sum(min(i+1,width)*n for i,n in enumerate(hist))/sum(hist)
        rows.append({'verify_width':width,'expected_committed_from_histogram':expected,'cycle_us':cost[width],'relative_committed_rate':expected/mean*cost[8]/cost[width]})
    return {'baseline_mean':mean,'rows':rows,'best_measured_width':max(rows,key=lambda r:r['relative_committed_rate'])['verify_width'],'limitations':'Empirical same-prefix counterfactual only; policy changes future state/distribution, so requires full-model validation. No adaptation performed.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--measurement');a=p.parse_args()
    if a.measurement:
        with open(a.measurement,encoding='utf-8') as f:print(json.dumps(analyze(json.load(f)),indent=2))
    else:
        print(json.dumps({'status':'NO_CYCLE_COST_OR_HISTOGRAM_MEASUREMENTS','mean_only_prefix_bounds':{str(mean):{str(k):bounds(mean,k) for k in (2,4,6)} for mean in (4.0,4.1)},'policy_selected':None},indent=2))
