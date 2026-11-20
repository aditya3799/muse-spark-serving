"""CPU arithmetic domain proof for the isolated50800 full-coefficient variant."""
import json,hashlib,pathlib,numpy as np
P=pathlib.Path(__file__).resolve().parent;D=P/'pr50800_boundaries';s=(D/'device_only.cu').read_text();assert '0x3B003B00u' in s and '0x7E807E80u' in s
assert 'const float sl = __fdividef(gv, 1.f + __expf(-gv));' in s
assert s.count('__bfloat162float(__float2bfloat16_rn(acc[b][')==8

def bf(x):
 a=np.asarray(x,np.float32);u=a.view(np.uint32)
 return np.asarray((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000),np.uint32).view(np.float32)
def fbits(bits):return np.asarray(bits,np.uint32).view(np.float32)
vals=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
rows=[];errors=[];subnormal=0;signedzero=0
for scale in range(127):
 e,m=scale>>3,scale&7;f=np.float32(m*2**-9 if e==0 else (1+m/8)*2**(e-7))
 p=2 if m>=4 else 1 if m>=2 else 0;encoded=p*8+((m<<(3-p))&7) if e==0 else (e+2)*8+m
 if scale==0:encoded=0
 sf=fbits((encoded*16+(118<<7))<<16)
 for code in range(16):
  used=0 if scale==0 else code # Exact existing upstream pack, not a hypothetical new pack.
  tiny=fbits((((used&7)<<6)|((used&8)<<12))<<16)
  full=bf(tiny*np.float32(2**126));got=bf(full*sf);want=np.float32(vals[code]*f)
  same=int(got.view(np.uint32))==int(want.view(np.uint32))
  if not same:errors.append(dict(scale=scale,code=code,got_bits=int(got.view(np.uint32)),expected_bits=int(want.view(np.uint32))))
  if scale in range(1,8):subnormal+=int(same)
  if code==8 and scale!=0:signedzero+=int(same)
  rows.append(same)
assert len(errors)==8 and all(r['scale']==0 and r['code']>=8 for r in errors)
assert subnormal==112 and signedzero==126
# Prove correction of the two formerly exposed finite boundary/range examples.
examples=[]
for a,g in [(1.00390625,.13),(2**-60,2**70),(2**-100,1.),(-0.,1.)]:
 parent=bf(bf(np.float32(a))*bf(np.float32(g)))
 variant=bf(np.float32(bf(np.float32(a)))*np.float32(bf(np.float32(g))))
 assert parent.view(np.uint32)==variant.view(np.uint32)
 examples.append(dict(accumulator=a,scale=g,output_bits=int(parent.view(np.uint32))))
r=dict(source_sha256=hashlib.sha256((D/'device_only.cu').read_bytes()).hexdigest(),pairs=2032,nonzero_positive_scale_pairs_exact=2016,subnormal_scale_pairs_exact=subnormal,negative_zero_code_nonzero_scale_pairs_exact=signedzero,zero_scale_sign_losses_from_unchanged_pack=errors,unscaled_coefficient_range=[2**-10,2688.],fixed_boundary_examples=examples,device_only=True,host_contract_present=False,limitations=['FP32 CPU emulation of BF16 arithmetic; not SM90 numerical execution.','Eight signed-zero coefficient differences remain inherited from original pack; no bitwise-all-codes claim.','Negative SF including negativezero and NaN are rejected by upstream pack, not tested as accepted.','SiLU source now removes extra BF16 round, but CUDA intrinsic/compiler parity needs actual native comparison.','Full-FFN order, normalization, top-k and tokens are not tested.'])
(D/'coefficient_proof.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
