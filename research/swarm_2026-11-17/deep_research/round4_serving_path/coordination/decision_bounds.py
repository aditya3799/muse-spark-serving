"""Bounds from acceptance means; deliberately not a serving predictor."""
from fractions import Fraction
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def extreme_distributions(mean,lo=1,hi=8):
    rows=[]
    for x in range(lo,hi+1):
        for y in range(x,hi+1):
            if not x<=mean<=y: continue
            if x==y:
                if mean==x: rows.append(((x,Fraction(1)),))
            else:
                py=(mean-x)/(y-x)
                rows.append(((x,1-py),(y,py)))
    return rows

def expectation(dist,fn):return sum(p*fn(x) for x,p in dist)

def bounds(mean,width):
    dist=extreme_distributions(mean)
    values=[expectation(d,lambda x:min(x,width)) for d in dist]
    lower=1+(width-1)*(mean-1)/7
    upper=min(mean,width)
    assert min(values)==lower and max(values)==upper
    # Second segment needed only when first width target predictions all match.
    # L includes the bonus, so survivor iff L > width, not L >= width.
    if width<8:
        prob=[expectation(d,lambda x:int(x>width)) for d in dist]
        p_low=max(Fraction(0),(mean-width)/(8-width))
        p_high=min(Fraction(1),(mean-1)/width)
        assert min(prob)==p_low and max(prob)==p_high
    else:p_low=p_high=Fraction(0)
    return {'width':width,'capped_commit_mean_lower':float(lower),'capped_commit_mean_upper':float(upper),
            'second_segment_survival_probability_lower':float(p_low),'second_segment_survival_probability_upper':float(p_high)}

baseline=Fraction(12237,10000);target=Fraction(5,2);factor=target/baseline
result={
 'scope':'Mathematical non-identifiability and decision gates; no hardware or serving prediction',
 'known_score':float(baseline),'target_score':float(target),
 'uniform_additional_rung_speedup_needed':float(factor),
 'uniform_fixed_work_runtime_reduction_needed':float(1-1/factor),
 'only_one_of_four_equal_log_weight_rungs_additional_factor_needed':float(factor**4),
 'score_limits':['Score is geometric mean of frontier ratios, not one aggregate runtime.',
                 'Runtime interpretation assumes identical work and uniform improvement on all four rungs.',
                 'A single affected phase with time fraction p and local speedup s yields only 1/(1-p+p/s) for that fixed workload.',
                 'Local stage gains must not be multiplied into the existing score without measured end-to-end rungs.'],
 'acceptance_mean_examples':[{'mean':float(mean),'bounds':[bounds(mean,w) for w in [1,2,4,6,8]]}
                             for mean in [Fraction(4),Fraction(41,10)]],
 'acceptance_limits':['L is total committed/generated tokens per ordinary width8 step, including bonus, supported on1..8.',
                      'Published acceptance means are rounded and do not identify the histogram or conditional confidence.',
                      'Capped means refer to clipping the SAME existing proposals at the SAME boundary. Redrafting changes future proposals and their distribution.',
                      'Segment survival bounds require shape-invariant target predictions and no terminal token truncation.',
                      'No independence or geometric acceptance assumption is used.'],
 'candidate_decision_gates':{
   'capped_prefix':'Measure own-policy committed tokens divided by full step time D8+T(W)+aux(W)+control(W), then validate full benchmark. Compare same batch/context/graph regimes; do not assume T(W) proportional toW.',
   'segmented_same8':'Before a continuation controller is justified, measure T4(B,L)+T4(S,L+4)+decision+compaction+extra_aux against T8(B,L). Same proposals/output length cancel only after complete equivalence qualification.',
   'attention_workspace':'Compare complete projection+collective+norm with matched PDL policy/current comm. A producer-only delta cannot establish complete path benefit.',
   'captured_argmax':'Compare full verify/accept tail with realTP8, exactbucket coverage and non-greedy overhead. Collective emulation only checks lifecycle/arithmetic.',
   'scheduler':'Reordering requires measured service-overlap or contention benefit; additive-cost makespan is unchanged. Rank-local wallclock policies need coordinated collective order.'},
 'source_notes':[
  'Scheduler metrics_reporter.update_spec_metrics uses generatedtokens/forwardrequestcount; batch_result_processor.process_batch_result_decode supplies generatedtokens.',
  'Only existing saved scored artifacts supply1.2237; no new submission performed.'
 ]}
result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(HERE/'DECISION_BOUNDS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'additional_factor':float(factor),'uniform_time_reduction':float(1-1/factor),'mean4.1_width4':bounds(Fraction(41,10),4)},indent=2))
