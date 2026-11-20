"""A full-width cross-shard reduction counterexample using normal BF16 values."""
import json
import math
import struct
from pathlib import Path
here=Path(__file__).resolve().parent
f32=lambda x:struct.unpack("<f",struct.pack("<f",x))[0]
def b16(x):
    bits=struct.unpack("<I",struct.pack("<f",x))[0]
    return struct.unpack("<f",struct.pack("<I",((bits+32767+((bits>>16)&1))>>16)<<16))[0]
# The full Triton layout combines corresponding components across the 2048
# feature boundary before it combines either thread with the neighboring lane (feature offset 8).
b=(1.,0.)
rows=[]
for a in ((4096.,1.),(1.,4096.)):
    parent=f32(f32(a[0]*a[0]+b[0]*b[0])+f32(a[1]*a[1]+b[1]*b[1]))
    separated=f32(f32(a[0]*a[0]+a[1]*a[1])+f32(b[0]*b[0]+b[1]*b[1]))
    gate=.505859375
    def rounded(s):
        inv=f32(1./math.sqrt(f32(f32(s/4096)+1.e-8)))
        return [inv,b16(f32(gate*inv))]
    rows.append(dict(left=a,right=b,full_square_sum=parent,scalar_square_sum=separated,
        correctly_rounded_reciprocal_and_bf16_output=rounded(parent),scalar_reciprocal_and_bf16_output=rounded(separated)))
assert rows[0]["full_square_sum"]!=rows[1]["full_square_sum"]
assert rows[0]["scalar_square_sum"]==rows[1]["scalar_square_sum"]
assert rows[1]["correctly_rounded_reciprocal_and_bf16_output"]!=rows[1]["scalar_reciprocal_and_bf16_output"]
result=dict(rows=rows,score=.505859375,weight=1,scale=1,hidden_size=4096,active_route_features=[0,8,2048,2056],
    scope="FP32 squared-sum impossibility for independent shard scalar totals. CPU output models correctly rounded reciprocal square root; actual GPU rsqrt.approx and layout must be independently checked.")
(here/"counterexample.json").write_text(json.dumps(result,indent=2))
print(rows)
