"""Independent CPU arithmetic/bit audit of pinned SGLang PR50800; no external code execution."""
import ast,hashlib,json,math,pathlib,numpy as np
P=pathlib.Path(__file__).resolve().parent; S=P/'sources/nov10_sweep/pr50800'
cu=(S/'nvfp4_w4a16_sm90.cuh').read_text(); rp=(S/'nvfp4_sm90_repack.py').read_text()
assert hashlib.sha256(cu.encode()).hexdigest()=='92219b6456049529db50af189887b7a17132b3ce53a4375bf87912ea0f19f572'
pos=next(ast.literal_eval(n.value) for n in ast.parse(rp).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_POS' for t in n.targets))
def bf(x):
 a=np.asarray(x,dtype=np.float32);u=a.view(np.uint32);return np.asarray((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000),dtype=np.uint32).view(np.float32)
def tiny(c):return ((c&7)<<6)|((c&8)<<12)
def pack(c):
 q=0
 for m in range(4):
  for h in range(2):
   for b,p in enumerate(pos[m][h]):q|=((int(c[4*h+m])>>b)&1)<<p
 return q
def decode(q):
 mask=0x81c081c0;r=((q<<6)|(q>>26))&0xffffffff
 return [q&mask,(q<<3)&mask,(q<<6)&mask,((r&0x001b001b)*0x1084)&mask]
rng=np.random.default_rng(930383);vectors=[rng.integers(0,16,8) for _ in range(65536)]
for lane in range(8):
 for c in range(16):
  a=np.zeros(8,dtype=np.int32);a[lane]=c;vectors.append(a)
for c in vectors:
 got=decode(pack(c));expected=[tiny(int(c[m]))|(tiny(int(c[4+m]))<<16) for m in range(4)]
 assert got==expected
fp=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
checked=0;zero_sign=[]
for s in range(127):
 e,m=s>>3,s&7;sf=np.float32(m*2**-9 if e==0 else (1+m/8)*2**(e-7))
 p=2 if m>=4 else 1 if m>=2 else 0
 code=(p*8+((m<<(3-p))&7)) if e==0 else (e+2)*8+m
 if s==0:code=0
 sb=np.array([(code*16+(181<<7))<<16],np.uint32).view(np.float32)[0]
 for c in range(16):
  used=0 if s==0 else c
  t=np.array([tiny(used)<<16],np.uint32).view(np.float32)[0]
  got=bf(np.float32(t*sb));target=np.float32(fp[c]*sf*np.float32(2**-63))
  assert float(got)==float(target)
  if got.view(np.uint32)!=target.view(np.uint32):zero_sign.append([s,c])
  checked+=1
# Global scaling placement: equal global gate/up, ordinary finite normalized-sized values.
a=np.float32(1.00390625);g=np.float32(.13)
parent=bf(bf(a)*bf(g));donor=bf(a*g)
# Additional SiLU rounding changes the operation even with identical GEMM/scaling.
silu_example=None
for x in bf(np.linspace(-6,6,12001,dtype=np.float32)):
 for up in (np.float32(.1171875),np.float32(.9765625),np.float32(3.125)):
  sl=np.float32(x/(1+math.exp(-float(x))))
  old=bf(sl*up);new=bf(bf(sl)*up)
  if old.view(np.uint32)!=new.view(np.uint32):silu_example=dict(gate=float(x),up=float(up),unrounded_activation_result=float(old),rounded_activation_result=float(new));break
 if silu_example:break
with np.errstate(over='ignore',invalid='ignore',under='ignore'):
 tinyx=np.float32(2**-100); unscaled=np.float32(tinyx*1); scaled=np.float32(tinyx*np.float32(2**-63));restored=np.float32(scaled*np.float32(2**63))
 assert restored==0 and unscaled!=0
 gs=np.float32(2**70); acc=np.float32(2**-60); parent_ext=bf(bf(acc)*bf(gs)); donor_ext=bf(np.float32(acc*np.float32(2**-63))*np.float32(gs*np.float32(2**63)))
assert parent.view(np.uint32)!=donor.view(np.uint32)
r=dict(source_sha256=hashlib.sha256(cu.encode()).hexdigest(),source_head='81502f72823876ef01c1f273d7061e1705361735',decode_words_checked=len(vectors),coefficient_pairs_checked=checked,positive_finite_scale_numeric_exact=True,zero_sign_differences=zero_sign,negative_and_nan_scale_domain='Upstream rejects all sign-bit scales, including -0, and NaNs; no support claimed.',global_boundary_example=dict(accumulator=float(a),global_scale=float(g),scored_parent=float(parent),upstream=float(donor)),silu_rounding_example=silu_example,exponent_offset_fp32_underflow=dict(input=float(tinyx),unscaled_product=float(unscaled),scaled_then_restored=float(restored)),exponent_offset_global_overflow=dict(accumulator=float(acc),global_scale=float(gs),scored_parent=float(parent_ext),upstream=str(float(donor_ext))),limits=['CPU equations only; does not execute PTX or tensor cores.','FP32 exponent-offset counterexamples are exceptional-range feasibility limits, not evidence real normalized Muse inputs encounter them.','Changed boundaries do not alone decide legality/quality; full-model gate remains required.','Not a benchmark, H200 runtime result, or scored-parent speedup.'])
(P/'pr50800_coefficient_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

