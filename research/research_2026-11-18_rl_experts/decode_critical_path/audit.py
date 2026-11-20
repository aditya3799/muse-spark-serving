"""Aggregate H200 whole-layer traces at their own 80K benchmark scope."""
import json
import re
from pathlib import Path
import hashlib
here=Path(__file__).resolve().parent
root=here.parents[1]
public=root/"references/gauravjain14-muse-okc-submissions"
d01=public/"muse_decode_layer/01_public_pipeline_bo21L_dense/evidence/d01_nsys.txt"
ncu=d01.with_name("d01_analysis.txt")
p81=root.parent/"muse_decode_layer/profiles/submission.725c19cbeb9b4d48.json"
run=json.loads(p81.read_text(encoding="utf-8"))
def parse_p81():
    by={}
    for p in run["profiles"]:
        case=p["case"];note=p["report"]
        span=re.search(r"GPU busy[^:\n]*:\s*([\d.]+)",note)
        lines=note.splitlines()
        total=float(re.search(r"^total\s+([\d.]+)",note,re.M)[1])
        attn=float(next(re.search(r"\s([\d.]+)\s+µs",s)[1] for s in lines if "::attn_kernel<" in s))
        busy=[s for s in lines if "GPU busy" in s]
        if busy:
            busy_match=re.search(r"([\d.]+)\s+µs",busy[0])
            if not busy_match:raise ValueError(busy[0])
            total=float(busy_match[1])
        by[(case["load"],case["layer"])]=(total,attn)
    return by
def parse_d01():
    by={}
    for i,span,body in re.findall(r"CASE (\d+): span ([\d.]+) us, \d+ samples\n(.*?)(?=CASE \d+:|\Z)",d01.read_text(),re.S):
        p=run["profiles"][int(i)]["case"]
        attn=float(re.search(r"([\d.]+) us\s+attn_kernel #",body)[1])
        by[(p["load"],p["layer"])]=(float(span),attn)
    return by
tables={}
for name,by in [("public_d01_nsys",parse_d01()),("team_s081_profile",parse_p81())]:
    rows={}
    for load in ["decode 1","decode 16","decode 32","verify 1","verify 16","verify 32"]:
        total=sum(n*by[load,w][0] for w,n in [("sliding",46),("full",16)])
        attn=sum(n*by[load,w][1] for w,n in [("sliding",46),("full",16)])
        full=16*by[load,"full"][1]
        rows[load]=dict(layer_weighted_us=round(total,3),attention_us=round(attn,3),attention_fraction=round(attn/total,5),full_attention_us=round(full,3),
           zero_attention_upper=round(total/(total-attn),4),zero_full_attention_upper=round(total/(total-full),4))
    tables[name]=rows
floor={}
for num,name in [(9,"verify 16"),(11,"verify 32")]:
    sec=re.search(rf"CASE {num}:.*?\n(.*?)(?=CASE |\Z)",ncu.read_text(),re.S)[1]
    m=re.search(r"attn_kernel\s+([\d.]+) us\s+([\d.]+) MB",sec)
    us=float(m[1]);mb=float(m[2])
    # The decoder board's roofline specifies 4.8 TB/s. Hold measured HBM bytes fixed.
    min_us=mb/4.8
    gain=us-min_us
    tot=tables["public_d01_nsys"][name]["layer_weighted_us"]
    floor[name]=dict(full_kernel_measured_us=us,dram_MB=mb,fixed_bytes_min_us=round(min_us,3),save_us_per_full_layer=round(gain,3),full_layer_count=16,conditional_target_step_upper=round(tot/(tot-16*gain),4))
out=dict(scope="Single-rank operational H200 kernel-board proxy, 80K page1; TP8/EP8 collectives are identity, no scheduler, sampler, DFlash, interconnect or host. Eliminating listed time assumes other operations retain original runtime and overlap. Profile rounding and independently summarized medians limit precision. No page-entry or sampler allocation from these fused or absent donor regions.",files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (p81,d01,ncu)},tables=tables,fixed_bytes_HBM_upper=floor)
(here/"h200.json").write_text(json.dumps(out,indent=2))
print(json.dumps(tables["public_d01_nsys"],indent=1));print(floor)
