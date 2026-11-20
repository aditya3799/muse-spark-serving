"""Summarize paired full-envelope receipts. Positive gain is baseline/candidate >1."""
import json
import argparse
import statistics
from pathlib import Path
import numpy as np
F=Path(__file__).resolve().parent
median=statistics.median
def summarize(directory):
    j=json.loads((F/directory/"canonical.json").read_bytes())
    assert j["correct"] and j["complete"] and j["active"]
    rows={}
    for name,v in j["bench"]["shapes"].items():
        t=v["times"];a,b,c=map(np.asarray,(t["baseline"],t["candidate"],t["control"]))
        assert len(a)==len(b)==len(c)==8 and v["order"]==["AB","BA"]*4
        ra=a/b;rc=a/c
        rng=np.random.default_rng(j["seed"]+len(name))
        ix=rng.integers(0,len(ra),size=(10000,len(ra)))
        rows[name]=dict(actual_native=v["native_boundary_active"],execution=v["execution"],
            chunks=v["chunks"],baseline_us=median(a),control_us=median(c),candidate_us=median(b),
            paired_gain=median(ra),control_ratio=median(rc),control_median_absolute_deviation_from_1=median(abs(rc-1)),
            gain_95_percentile_bootstrap=np.quantile(np.median(ra[ix],axis=1),[.025,.975]).tolist(),
            absolute_gain_us=median(a-b),paired_samples=8,all_bitwise=all(v["parity"]))
    return rows
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--ffn",action="store_true");parser.add_argument("--bucket",action="store_true")
    mode=parser.parse_args()
    prefix="bucket_ffn" if mode.bucket else "ffn" if mode.ffn else "ballot"
    train=summarize("bucket_ffn_train" if mode.bucket else "ffn_train_exclusive" if mode.ffn else "ballot_train")
    test=summarize(prefix+"_holdout")
    assert train.keys()==test.keys()
    out=dict(train=train,holdout=test,interpretation="Full local rank Humming W13/W2/finalize envelope only; separate train/holdout seeds. 16K cases are charged four real M4096 graph calls and input staging. Date-dependent laptop power mode, target communication/attention/prefill scheduling omitted. Percentile resampling of eight local pairs is descriptive and does not calibrate H200 serving.")
    if mode.ffn:
        out["interpretation"]="Rank-local two dense projections, weighted RMSNorm, Humming W13/W2 and per-expert RMS/weighted sum. First training receipt excluded due to possible warmup process overlap. No router, attention, post-FFN residual, or collectives. 753/3012 represent exact-size judge eager fallback although the local pairing uses CUDA graphs; 3016 is an aligned sensitivity with an actual rank0 377-row shard. Normally admitted judge prefill replay rounds 753/3012 to 768/3072, measured separately in bucket_ffn_summary.json. Peer rows staged outside captured region and expert all-reduce identity. Synthetic inputs and eight paired AB/BA repeats per seed."
    if mode.bucket:
        out["interpretation"]="Matched bucket-shape rank-local FFN replay on RTX3050. Humming W13/W2 + RMS/weighted sum; 8192?4096 pooler, pre-expert weighted RMS, 4096?8192 unpooler. M768 and M3072 compute actual local 96/384 row full-K projections. Peer rows prepared before graph, identity at EP sum, no actual collectives/router/attention or residual. Routes synthetic for all bucket rows. No H200 per-request bucket or tail route trace."
        out["mode_evidence"]={
            "recorded": "Serving job 317c88a5fd474cd2 startup says target prefill CUDA graph capture ended after 151.72 s, 15.56 GB. Reported graph memory=15.559 GB. Startup log truncated/cut at first request. Actual per-request dispatch not exposed.",
            "buckets":"Pinned 297e54f. cuda_graph_hook.py:480-494 generates >=512 token buckets; memory_hook.py:232 selects default maximum from chunked-prefill size for GQA; prefill_cuda_graph_runner.py:1850 calls _pad_to_bucket, base_cuda_graph_runner.py:179 bisects to smallest fitting bucket. Match 753?768 and 3012?3072 when normal graph replay admitted. Capture-size caps, prefix eligibility, or mixed modes can select eager.",
            "rankwork":"Scored muse_spark_gemm_ag.py:74 requires M>=512 and M%world=0; local FP32-output matmul ? BF16 then row gather. Each synthetic paired call at 768/3072 runs its rank-0 96/384-row projections; mock peers are staged inputs, not a collective.",
            "eager_shapes":"Original ffn_summary.json at 753/3012 models exact eager fallback inputs. Its M3016 case is an aligned sensitivity, not the normal 3012 graph bucket."}
    (F/(prefix+"_summary.json")).write_text(json.dumps(out,indent=2)+"\n")
    for k,p in test.items():
        if p["actual_native"]:
            print(k,"base/ctrl/cand ms",*[round(p[key]/1000,3) for key in ("baseline_us","control_us","candidate_us")],"gain",round(p["paired_gain"],4),"CI",[round(x,4) for x in p["gain_95_percentile_bootstrap"]],"AA",round(p["control_ratio"],4),"ms gained",round(p["absolute_gain_us"]/1000,3))
if __name__=="__main__":main()

