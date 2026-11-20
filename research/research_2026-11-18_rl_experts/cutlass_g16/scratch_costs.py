"""Screen complete-rank sequential FP8 slab allocation and exact nibble expansion."""
import json
from pathlib import Path
import sys
d=Path(__file__).resolve().parent
sys.path.insert(0,str(d.parent/"upstream_scale_order"))
from cpu import weight,scale,bits,bf
values={bits(scale(b)):b for b in range(256) if b not in (127,255)}
codewords=[values[bits(weight(q))] for q in range(16)]
for q in range(16):
    for s in range(256):
        if s in (127,255):
            continue
        parent=bf(bf(bf(weight(q)*2.**-126)*2.**124)*bf(bf(scale(s)*2.**-120)*2.**120))
        temporary=bf(bf(scale(codewords[q]))*bf(scale(s)*.25))
        assert bits(parent)==bits(temporary),(q,s,parent,temporary)
layers=60
n=32
k=4096
cases={}
for stage,width in (("w13",8192),("w2",4096)):
    elems=n*k*width
    # Scored source holds packed FP4 plus E4M3 scales; generic candidate
    # materializes decoded FP8 base weights and converted pre-divided BF16 scales.
    original=elems//2+elems//16
    scratch=elems+2*(elems//16)
    cases[stage]=dict(expanded_base_weight_bytes=elems,expanded_bf16_scale_bytes=2*(elems//16),
            source_bytes=original,scratch_bytes=scratch,read_write_repack_bytes=original+scratch,
            additional_traffic_including_one_read_of_scratch_over_one_original_read=2*scratch)
tot=sum(v["read_write_repack_bytes"] for v in cases.values())
out=dict(fully_resident_banks=True,local_experts=32,all_finite_operands=16*254,
    e2m1_to_e4m3_bytes=codewords,stages=cases,sequential_peak_bytes=max(v["scratch_bytes"] for v in cases.values()),
    total_repack_bytes_per_layer=tot,total_repack_bytes_60_layers=tot*layers,
    placeholder_bandwidth_Bps=4.8e12,conditional_seconds_for_repack_60_layers=tot*layers/4.8e12,
    exclusions="No route rows, activation matrices, GEMM outputs, scales originally pinned for fallback, TMA descriptors, launches, cache behavior or reuse across batches. The 4.8 TB/s division is an idealized data-motion floor if all counted bytes cross HBM, not a serving latency estimate.",
    alternative_exact_same_byte_count="Store four FP8 scales in a 32-bit word per output/64 K. Input remains FP4 if its fragment conversion is hardware-qualified.")
(d/"scratch_costs.json").write_text(json.dumps(out,indent=2))
print("peak MiB",out["sequential_peak_bytes"]/2**20,"read+write MiB/layer",tot/2**20,"all layers ideal ms",1e3*tot*layers/4.8e12)
