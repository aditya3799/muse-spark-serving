"""Sensitivity analysis; no claim that these shape traces match the challenge.

Costs are interpolated single-GPU measurements. NCCL and CPU-vote additions are
explicit hypothetical parameters. Every failed/unpublished screen is charged.
"""
import hashlib,json,math,random,statistics
from pathlib import Path
from sightings_policy import SightingsPolicy
HERE=Path(__file__).resolve().parent
source=HERE/'ragged_n_pilot.json';pilot=json.loads(source.read_text())
points=sorted((c['rows'],c) for c in pilot['cases'] if c['qualified'] and c['under128MiB_transport_calibration'])

def costs(m):
    left=max((p for p in points if p[0]<=m),default=points[0]);right=min((p for p in points if p[0]>=m),default=points[-1])
    fraction=0 if left[0]==right[0] else (m-left[0])/(right[0]-left[0])
    def value(item,name):
        if name=='native':return min(item['medians_us']['baseline'],item['medians_us']['control'])
        if name=='candidate':return item['medians_us']['candidate']
        return item['first_use_extra_qualification_us']
    return {name:value(left[1],name)*(1-fraction)+value(right[1],name)*fraction for name in ('native','candidate','first_extra')}

def sequences(seed=417631,length=416):
    rng=random.Random(seed);keys=list(range(512,1536));rng.shuffle(keys)
    result={'unique':keys[:length]}
    for repeat in (2,4,8,16,32):
        trace=(keys[:math.ceil(length/repeat)]*repeat)[:length];rng.shuffle(trace)
        result[f'shuffled_{repeat}_hits']=trace
    result['one_hot']=[keys[0]]*length
    result['zipf_1_1']=rng.choices(keys,weights=[1/(i+1)**1.1 for i in range(len(keys))],k=length)
    result['bursty_four_then_abandon']=[key for key in keys[:length//4] for _ in range(4)]
    result['bursty_eight_then_abandon']=[key for key in keys[:length//8] for _ in range(8)]
    return result

def run(trace,threshold,*,bandwidth_gbps=None,cpu_vote_us=0,failure_fraction=0):
    policy=SightingsPolicy(min_sightings=threshold,metadata_capacity=64,plan_capacity=8)
    native=0.;candidate=0.;extra=0.;saved=0.;steps={};per_shape={}
    for m in trace:
        c=costs(m);native+=c['native'];decision=policy.visit((m,40960));steps[decision]=steps.get(decision,0)+1
        transfer=0 if bandwidth_gbps is None else (7*m*1024*2)/(bandwidth_gbps*1000)
        cost=c['native']
        if decision=='fast':cost=c['candidate']+transfer;saved+=c['native']-cost
        elif decision=='screen':
            # Includes all eight local shard comparisons measured in the pilot.
            # Add one actual-transport check plus explicit CPU coherence cost.
            add=c['first_extra']+c['candidate']+transfer+cpu_vote_us
            cost+=add;extra+=add
            rejected=(int(hashlib.sha256(str(m).encode()).hexdigest()[:8],16)%10000)<failure_fraction*10000
            policy.outcome((m,40960),decision,not rejected)
        elif decision=='validate':
            add=c['candidate']+transfer+cpu_vote_us;cost+=add;extra+=add
            policy.outcome((m,40960),decision,c['candidate']+transfer<c['native'])
        candidate+=cost;per_shape[m]=per_shape.get(m,0)+1
    return dict(requests=len(trace),unique_shapes=len(per_shape),max_exact_shape_repeats=max(per_shape.values()),
        native_us=native,policy_us=candidate,component_speedup=native/candidate,
        qualification_extra_us=extra,realized_fast_saving_us=saved,steps=steps,
        active_shapes=len(policy.active),pending_unpublished_shapes=len(policy.pending),rejected_shapes=len(policy.rejected),
        metadata_entries=len(policy.seen),all_qualification_cost_charged=True)

report=dict(scope='Interpolated RTX3050 component-cost sensitivity, not actual serving traces or H200 prediction. Exact shape distribution is unknown.',
    pilot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),states='metadata-only sightings; screen+validate each returns native; fast only after both accept; no captured plan eviction',
    constraints=dict(metadata_capacity=64,plan_capacity=8,shape_domain=[512,1535],no_kernel_work_on_early_sightings=True),cases=[])
for name,trace in sequences().items():
    for threshold in (2,4,8,16,32):
        for scenario,bw,cpu,failure in (('local_optimistic',None,0,0),('hypothetical_25GBps_0.5ms_votes',25,500,0),('hypothetical_25GBps_0.5ms_votes_30pct_reject',25,500,.3)):
            report['cases'].append(dict(trace=name,threshold=threshold,scenario=scenario,**run(trace,threshold,bandwidth_gbps=bw,cpu_vote_us=cpu,failure_fraction=failure)))
(HERE/'amortization_model.json').write_text(json.dumps(report,indent=2)+'\n')
for name in sequences():
    row=[c for c in report['cases'] if c['trace']==name and c['scenario']=='local_optimistic']
    print(name,' '.join(f"s{c['threshold']}={c['component_speedup']:.3f}x({c['steps'].get('fast',0)}fast,{c['pending_unpublished_shapes']}pending)" for c in row),flush=True)
