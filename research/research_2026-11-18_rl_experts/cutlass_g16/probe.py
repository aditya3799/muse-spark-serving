"""Source and finite-value screen of Hopper mixed-input grouped GEMM for Muse."""
import hashlib
import json
import math
import sys
from pathlib import Path

d=Path(__file__).resolve().parent
r=d/"reference"
sys.path.insert(0,str(d.parent/"upstream_scale_order"))
from cpu import bits,bf,weight,scale

def check(file,*snippets):
    data=(r/file).read_text(encoding="utf8")
    assert all(needle in data for needle in snippets)
    return hashlib.sha256((r/file).read_bytes()).hexdigest()

checks={}
files={
"69_hopper_int4_bf16_grouped_gemm.cu":["using QuantType = cutlass::int4b_t","using ElementScale = cutlass::bfloat16_t","using ElementAccumulator  = float","constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value"],
"55_hopper_int4_bf16_gemm.cu":["using QuantType = cutlass::float_e2m1_t","CUTLASS_MIXED_DTYPE_E2M1"],
"sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp":["args.chunk_size == K || ((args.chunk_size % size<2>(TileShape{})) == 0)","size<1>(SmemLayoutAtomScale{}) == 1","const int scale_load_k = *k_tile_iter / mainloop_params.reload_factor","Utils::dequantize_A_kblock"],
"sm90_gmma_builder.inl":["detail::rs_smem_selector<GmmaMajorA, ElementAMma","detail::rs_smem_selector<GmmaMajorB, ElementBMma"],
"sm90_common.inl":["BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0"],
"mma_traits_sm90_gmma.hpp":["Layout<Shape<_8, _128>","upcast<sizeof_bits<Type>::value>(Layout_K_INTER_Atom_Bits{})"],
"mixed_input_utils.hpp":["auto stage = make_tensor_like<ElementScale>","stage(j) *= scales_vm(j, i)"],
"README.md":["group-wise scaling for identical problem shapes across all groups"]}
for file,anchors in files.items():
    checks[file]=check(file,*anchors)

# Original cutlass E4M3 staging multiplies in the scale's type when different from BF16.
# Emulate round-to-nearest-even/saturate-finite only for this numeric alternative.
finite=[i for i in range(256) if i not in (127,255)]
eb=[scale(i) for i in range(127)]
def fp8_round(x):
    val=abs(x)
    q=min(range(127),key=lambda i:(abs(eb[i]-val),i%2))
    return math.copysign(eb[q],x)

different=0
example=None
selectors=0
# Concrete storage-compatible proposal: encode four original E4M3 bytes in
# one 32-bit value at each N/K64 location. Decode exact lane at k_block 0..3.
for s in finite:
    quartet=[s,(s+37)%127,(s+83)%127,128+(s+29)%127]
    for ix in range(4):
        moved=quartet[ix:]+quartet[:ix]
        word=sum(b<<(8*j) for j,b in enumerate(moved))
        for j in range(4):
            actual=(word>>(8*j))&255
            assert actual==moved[j]
            for q in range(16):
                raw_weight=bf(weight(q)*2.**-126)
                raw_scale=bf(scale(actual)*2.**-120)
                parent=bf(bf(raw_weight*2.**124)*bf(raw_scale*2.**120))
                scaled=bf(bf(weight(q)/4)*bf(scale(actual)))
                assert bits(parent)==bits(scaled)
                selectors+=1
    for q in range(16):
        baseline=bits(bf(weight(q)*scale(s)/4.))
        staged=bits(bf(fp8_round(weight(q)*scale(s))/4.))
        if baseline!=staged:
            different+=1
            if example is None and weight(q)*scale(s)>1.:
                example=dict(q=q,scale_hex=hex(s),weight=weight(q),scale=scale(s),
                             parent_operand=bf(weight(q)*scale(s)/4.),
                             fp8_staged_operand=bf(fp8_round(weight(q)*scale(s))/4.))
full_k=4096
rows=[dict(cta_k=k,group16_source_guard=(16==full_k or 16%k==0),fp4_layout_k_atom=(k%32==0)) for k in (8,16,32,64,128)]
assert not any(x["group16_source_guard"] and x["fp4_layout_k_atom"] for x in rows)
n13,n2=8192,4096
experts=32
weight_bytes=experts*(n13+n2)*full_k//2
scale_bytes=experts*(n13+n2)*full_k//16
out=dict(upstream_sha="cd9d8a7d612b6b9ad0f1b698925e31243e828629",source_hashes=checks,cta_k=rows,
         frozen_full_k=full_k,minimum_k_inter_atom=128//4,parent_default_cta_k=64,
         finite_fp4_scale_pairs=16*len(finite),generic_e4m3_staging_counterexamples=different,
         illustrative_fp8_stage_counterexample=example,
         packed_k64_four_scales_per_word_selector_and_operand_comparisons=selectors,
         packed_draft_design="one unchanged 32-bit scale word per output channel per 64 K; select by K16 GMMA block, BF16 product with /4 weight, retain BF16 output rounding then x4 restoration",
         extra_weight_bytes_if_second_full_expert_layout_resident_per_layer=weight_bytes,
         extra_scale_bytes_if_second_scale_layout_resident_per_layer=scale_bytes,
         extra_combined_bytes_if_second_layout_60_layers=(weight_bytes+scale_bytes)*60,
         limit="No measured GEMM, no H200 WGMMA accumulator equivalence proof, no CUTLASS-Humming shared packed physical layout, no distributional timing or official gain.")
(d/"screen.json").write_text(json.dumps(out,indent=2))
print(different,selectors,example,weight_bytes,scale_bytes)
