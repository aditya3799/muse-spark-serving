"""Check frozen and BF16-native operands/offsets and derive finite static memory costs."""
import json
import math
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/"upstream_scale_order"))
from cpu import bf, bits, scale, weight
finite=[s for s in range(256) if s not in (127,255)]
for s in finite:
    canonical=bf(scale(s))
    decoded=bf(bf(scale(s)*2.**-120)*2.**120)
    assert bits(canonical)==bits(decoded)
    for q in range(16):
        fragment=bf(bf(weight(q)*2.**-126)*2.**124)
        assert bits(bf(fragment*canonical))==bits(bf(fragment*decoded))
total_orig, main_x, main_y=246,124,120
assert total_orig-main_x-main_y==126-124==2
w13=32*8192*4096//16
w2=32*4096*4096//16
res=dict(finite_signed_e4m3_encodings=len(finite),fp4_group_operand_bit_comparisons=len(finite)*16,
    exponent_offsets={"parent":[246,124,120,2],"candidate":[126,124,0,2]},
    original_scale_layout="An identical transpose/column permutation for either dtype in transform_humming_weight_scale, use_mxmma disabled.",
    finite_domain="The BF16 decoding calculation uses RNE/gradual underflow. The native GPU stage and full FFN still need to independently verify the compiler/hardware path.",
    persistent_delta_bytes_per_rank_layer={"w13":w13,"w2":w2,"both":w13+w2},
    conditional_60_layer_delta_bytes={"all_both":60*(w13+w2),"all_w2":60*w2,"both_first16":16*(w13+w2)},
    conditional_60_layer_budget_note="Additional 96 MiB for both stages per layer is net over existing resident E4M3. Rank-restricted expansion is globally synchronized only at weight preparation. No runtime cache or dynamic residency. Peak expansion briefly retains original scale tensor while constructing the new packed tensor. Other pre-existing canonical checkpoint aliases and graph capture temporaries are outside this accounting.")
(here/"proof.json").write_text(json.dumps(res,indent=2)+"\n")
print(json.dumps(res))
