"""Match the unchunked serial FFN holdout to its four-pair pilot and scored H200 context."""
import json
import statistics
from pathlib import Path
import numpy as np
F=Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_bytes())
def subset(t,limit):
    a,b,c=(np.asarray(t[k][:limit]) for k in ("baseline","candidate","control"))
    ratio=a/b
    return dict(pairs=len(a),baseline_us=statistics.median(a),candidate_us=statistics.median(b),
       control_us=statistics.median(c),paired_ratio=statistics.median(ratio),
       baseline_over_control=statistics.median(a/c),saved_us=statistics.median(a-b),
       paired_range=[min(ratio),max(ratio)],
       paired_median_omitting_largest_ratio=statistics.median(np.delete(ratio,np.argmax(ratio))))
def summary(j):
    assert j["active"] and j["complete"] and j["correct"]
    metrics={}
    for name,r in j["bench"]["shapes"].items():
        assert name in ("M16384_uniform","M16384_cluster") and r["chunks"]==1 and r["execution"]=="eager"
        assert r["native_boundary_active"] and all(r["parity"]) and r["order"]==["AB","BA"]*(len(r["order"])//2)
        metrics[name]=dict(first_four=subset(r["times"],4),all=subset(r["times"],len(r["order"])))
    mem={}
    for tag in ("original_or_control","candidate"):
        records=[r for r in j["serial_memory_per_measured_call"] if r["path"]==tag]
        mem[tag]={k:max(r[k] for r in records) for k in ("allocated_before","peak_allocated","peak_reserved")}
    return dict(seed=j["seed"],cases=metrics,memory_bytes=mem)
def main():
    pilot=summary(read(F/"full16k_serial/canonical.json"));holdout=summary(read(F/"full16k_holdout/canonical.json"))
    receipt=read(F.parent/"release_2026-11-18/scored_submission_status.json")
    r=receipt["result"]
    assert receipt["id"]=="317c88a5fd474cd2" and r["frontier"]["score"]==1.2363
    h200={str(v["concurrency"]):{k:v[k] for k in ("reps","tok_s_per_gpu","tok_s_per_user_p50","ttft_first_p90_s","ttft_later_p90_s")} for v in r["rungs"]}
    result=dict(pilot=pilot,holdout=holdout,
        local_measure="One unchunked M16384 local rank FFN forward per timing arm, H=8192, prepooler+pre-RMS, W13/W2+post-expert RMS/weighted sum, unpooler, eight defacto TP ranks with 2048 real rank0 pool/unpool rows. Peers' prepooler rows prepared outside timing. Expert all-reduce, attention, routing GEMM/overlap and post-FFN residual absent. PyTorch allocated/reserved include fixtures, weight pack and allocator cache, not full model.",
        official_parent=dict(id=receipt["id"],score=r["frontier"]["score"],source="release_2026-11-18/scored_submission_status.json",rungs=h200,
             absolute_stage_times="The scored H200 receipt has end-to-end rung measurements and startup capture timings. It has no per-expert, FFN, or steady-state per-layer timing."),
        timing_anomaly="The held-out clustered last pair had A=1799.5ms, B=599.6ms and duplicate control=650.5ms. The pair is retained in all headline medians. Source of the outlier is unmeasured; use robust paired medians and record the sensitivity omitting the largest ratio.",
        scope="Unchunked serial eager FFN excludes graph replay. Matched local CUDA-graph bucket replay at 768/3072 is in bucket_ffn_summary.json. Raw H200 end-to-end rungs are distinct from local absolute baseline times.")
    (F/"full16k_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(holdout))
if __name__=="__main__":main()

