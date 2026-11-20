"""Disabled raw-original-scale device source; no compile, loader or runtime hook."""
import pathlib,hashlib,json,numpy as np
P=pathlib.Path(__file__).resolve().parent;D=P/'pr50800_rawscale';s=(P/'pr50800_boundaries/device_only.cu').read_text()
old='''__device__ __forceinline__ uint32_t sc_bf16(uint32_t v, uint32_t sel) {
  return prmt(v, 0, sel) * 16u + 0x3B003B00u;
}'''
new='''// Raw ORIGINAL E4M3 bytes, never upstream normalized scale codes.
// Runtime owner must certify finite scales before any capture/replay; no owner exists here.
// NaN maps to quiet NaN defensively, without a payload-parity claim.
__device__ __forceinline__ uint16_t original_scale_bf16(unsigned byte) {
  const unsigned magnitude = byte & 127u;
  const unsigned sign = (byte & 128u) << 8;
  if (magnitude == 127u) return uint16_t(sign | 0x7FC0u);
  unsigned bits = (magnitude << 4) + 0x3C00u;
  if (magnitude < 8u)
    bits = __bfloat16_as_ushort(__float2bfloat16_rn(float(magnitude) * 0x1p-9f));
  return uint16_t(bits | sign);
}
__device__ __forceinline__ uint32_t sc_bf16(uint32_t v, uint32_t sel) {
  (void)sel; // All three source call sites use0x4140 (low two original scale bytes).
  return unsigned(original_scale_bf16(v & 255u)) |
         (unsigned(original_scale_bf16((v >> 8) & 255u)) << 16);
}'''
assert s.count(old)==1 and s.count('sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140)')==3
s=s.replace(old,new)
s=s.replace('// Positive finite E4M3 scales only, matching upstream repacker domain.','// Finite original E4M3 scales of either sign; owner finite-certificate REQUIRED.')
s=s.replace('// Upstream pack erases signs at zero scale; no claim of original-code signed-zero parity.','// Raw lossless layout retains original FP4 codes and scale-zero signs; not upstream pack.')
s=s.replace('// Two block scales (bytes0,1 of v) -> full-value BF16 scale bits, each in both halves. The pack stores each E4M3\n// scale losslessly as the code 8 E + m with scale = 1.m * 2^(E - 9) (subnormals normalized), so its BF16 bits are\n// code *16 +(118 <<7) in this unshifted variant.','// Two original raw E4M3 bytes -> full-value BF16 scales, including signedzero/subnormals.')
(D/'device_only_uncompiled.cu').write_bytes(s.encode())

def bf(x):
 u=np.asarray(x,np.float32).view(np.uint32)
 return np.asarray((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000),np.uint32).view(np.float32)
def fbits(x):return np.asarray(x,np.uint32).view(np.float32)
vals=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
count=0;negative=0;zeros=0;subnormals=0
for byte in range(256):
 mag=byte&127
 if mag==127:continue
 e,m=mag>>3,mag&7;expected_sf=np.float32((m*2**-9 if e==0 else (1+m/8)*2**(e-7))*(-1 if byte&128 else 1))
 bits=int(bf(np.float32(mag*2**-9)).view(np.uint32))>>16 if mag<8 else (mag<<4)+0x3c00
 sf=fbits((bits|((byte&128)<<8))<<16);assert sf.view(np.uint32)==expected_sf.view(np.uint32)
 for code in range(16):
  tiny=fbits((((code&7)<<6)|((code&8)<<12))<<16)
  got=bf(bf(tiny*np.float32(2**126))*sf);want=np.float32(vals[code]*expected_sf)
  assert got.view(np.uint32)==want.view(np.uint32),(byte,code)
  count+=1;negative+=int(bool(byte&128));zeros+=int(mag==0);subnormals+=int(0<mag<8)
r=dict(source_sha256=hashlib.sha256((D/'device_only_uncompiled.cu').read_bytes()).hexdigest(),boundary_parent_sha256=hashlib.sha256((P/'pr50800_boundaries/device_only.cu').read_bytes()).hexdigest(),accepted_finite_coefficient_pairs=count,bitwise_exact_cpu=True,negative_scale_pairs=negative,signedzero_scale_pairs=zeros,subnormal_scale_pairs=subnormals,nan_scale_values_preserved_by_layout_but_not_qualified=[127,255],compiler_run=False,gpu_execution=False,runtime_guard_implemented=False,owner_requirement='Future canonical resident finite-scale lease must be certified once and invalidated before any raw/.data/reload write or graph mutation; current source is not callable.',changed_device_functions=['original_scale_bf16','sc_bf16'],all_other_boundary_variant_device_math_and_scheduler_unchanged=True)
(D/'coefficient_proof.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
