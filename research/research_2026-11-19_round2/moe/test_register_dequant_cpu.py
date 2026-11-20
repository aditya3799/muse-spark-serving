"""Exact finite BF16 dequantization proof, independent from the native lookup table."""
import hashlib
import json
from pathlib import Path
import numpy as np
here=Path(__file__).resolve().parent;root=here.parents[1]
fp4=np.array([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],np.float32)
sf=np.arange(256,dtype=np.uint32)
e=(sf>>3)&15;m=sf&7
scale=np.where(e==0,m.astype(np.float32)*np.float32(2**-9),(1+m.astype(np.float32)/8)*np.exp2(e.astype(np.float32)-7)).astype(np.float32)
scale=np.where(sf&128,-scale,scale).astype(np.float32)
scale[(sf&127)==127]=np.nan
reference=(scale[:,None]*fp4[None,:]).astype(np.float32)
finite=(sf&127)!=127
# Integer BF16 construction; no floating point multiplication is used to decode codes/scales.
normal=((sf&127)<<4)+0x3C00
small=np.array([0,0x3B00,0x3B80,0x3BC0,0x3C00,0x3C20,0x3C40,0x3C60],np.uint32)
scale_bits=np.where(e==0,small[m],normal)|((sf&128)<<8)
q=np.arange(16,dtype=np.uint32);mag=q&7
q_bits=np.where(mag>=2,(mag<<6)+0x3F00,np.where(mag==1,0x3F00,0))|((q&8)<<12)
a=(scale_bits.astype(np.uint32)<<16).view(np.float32)
b=(q_bits.astype(np.uint32)<<16).view(np.float32)
assert np.array_equal(a[finite].view(np.uint32),scale[finite].view(np.uint32))
assert np.array_equal(b.view(np.uint32),fp4.view(np.uint32))
actual=(a[:,None]*b[None,:]).astype(np.float32)
assert np.array_equal(actual[finite].view(np.uint32),reference[finite].view(np.uint32))
assert np.all((reference[finite].view(np.uint32)&65535)==0)
# Packed dq4/normalization across every nibble choice plus random interference
# from all the other fields; independent scalar serving_code inverse.
rng=np.random.default_rng(930263)
words=rng.integers(0,2**32,1<<20,dtype=np.uint32)
M=np.uint32(0x81C081C0)
def rotate(x,s):return (x>>np.uint32(s))|(x<<np.uint32(32-s))
def eq2(x,target):
    return np.where((x&65535)==target,65535,0).astype(np.uint32)|((np.where((x>>16)==target,65535,0).astype(np.uint32))<<16)
def normalize(r):
    z=r&np.uint32(0x01C001C0)
    zeros=eq2(z,0);halves=eq2(z,0x40)
    return ((r+np.uint32(0x3F003F00))-(halves&np.uint32(0x00400040)))&~zeros | (r&np.uint32(0x80008000))
for R in range(4):
    if R<3:
        r=words&M if R==0 else rotate(words,3*R)&M
    else:r=((words&np.uint32(0x001B001B))*np.uint32(4228))&M
    normal=normalize(r)
    for H in range(2):
        if R<3:
            shift=(6+3*R+16*H)%32
            rotated=rotate(words,shift)
            code=(rotated&7)|((rotated>>6)&8)
        else:
            half=words>>np.uint32(16*H)
            code=((half>>4)&1)|((half&3)<<1)|(half&8)
        expected=q_bits[code]
        observed=(normal>>np.uint32(16*H))&65535
        assert np.array_equal(observed,expected),(R,H)
parent=json.loads((root/'release_2026-11-18/scored_submission_status.json').read_bytes())
native=json.loads((root/'research_2026-11-18_target_moe/native_package_02/scored_50a2f4d696224d6b.json').read_bytes())
comparison=[]
for a,b in zip(parent['result']['rungs'],native['result']['rungs']):
    assert a['concurrency']==b['concurrency']
    comparison.append(dict(concurrency=a['concurrency'],parent_generation_tps=a['tok_s_per_user_p50'],
      native_generation_tps=b['tok_s_per_user_p50'],native_over_parent=b['tok_s_per_user_p50']/a['tok_s_per_user_p50'],
      native_truncated=b['truncated'],parent_acceptance=a['acceptance'],native_acceptance=b['acceptance']))
report=dict(complete=True,finite_scale_code_products=254*16,exact_BF16_products=True,
            signed_zero_included=True,nan_scales='retain incumbent lookup table for NaN payload compatibility',
            random_packed_words=len(words),checked_packed_values=len(words)*8,
            source_pin='native_package_02/hopper_expert.cu serving_pair; original donor submission_v139_wgs_kperm3.py dq4/wgs_seg',
            serving_comparison=comparison,
            interpretation='Existing WGS serving generation metrics regressed too; separate-run metrics do not prove an isolated kernel cause. Removing LUT loads is a hypothesis, not a measured speedup.')
(here/'register_dequant_cpu.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
