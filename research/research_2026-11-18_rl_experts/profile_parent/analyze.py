"""Summarize measured stage fractions, instrumentation overhead, and arithmetic ceilings."""
import json
import statistics
from pathlib import Path
here=Path(__file__).resolve().parent
doc=json.loads((here/"timings.json").read_text())
assert doc["complete"]
out={}
categories={"W13":["w13"],"W2":["w2"],"scale_siLU":["w13_scaling","silu"],
            "post_finalize":["post_expert_norm_scale_and_sum"],"dense":["pool_project_cast_assemble","unpool_project_cast"],
            "pre_norm":["pre_expert_rms"],"route_prep":["alignment","partition"]}
med=statistics.median
for name,rec in doc["cases"].items():
    if not rec.get("complete"):
        out[name]={"complete":False,"reason":rec.get("oom","incomplete")};continue
    samples=rec["samples"]
    shares={k:med(sum(s["components_us"].get(op,0) for op in ops)/s["marked_stream_us"] for s in samples) for k,ops in categories.items()}
    gemm=med((s["components_us"]["w13"]+s["components_us"]["w2"])/s["marked_stream_us"] for s in samples)
    out[name]=dict(medians=rec["median_us"],median_components=rec["median_components_us"],shares=shares,
       local_pairs=rec["local_pairs"],active_experts=rec["active_experts"],local_route_share=rec["local_pairs"]/(rec["rows"]*8),
       top4_share=sum(sorted(rec["rank_histogram"],reverse=True)[:4])/rec["local_pairs"],
       measured_W13_W2_share=gemm, time_saved_2x_W13_W2=gemm/2,
       speedup_if_W13_W2_halved=1/(1-gemm/2), ceiling_if_W13_W2_removed=1/(1-gemm),
       unattributed_share=med(s["unattributed_us"]/s["marked_stream_us"] for s in samples),
       median_marker_to_plain=med(s["marked_stream_us"]/s["plain_stream_us"] for s in samples),
       sum_vs_plain=med(s["component_sum_us"]/s["plain_stream_us"] for s in samples))
    print(name,"parent ms",round(rec["median_us"]["plain_stream_us"]/1000,3),
          "components", {k:round(v*100,1) for k,v in shares.items()},
          "distortion",round(out[name]["median_marker_to_plain"],3),"halved",round(out[name]["speedup_if_W13_W2_halved"],3))
(here/"summary.json").write_text(json.dumps(out,indent=2)+"\n")


