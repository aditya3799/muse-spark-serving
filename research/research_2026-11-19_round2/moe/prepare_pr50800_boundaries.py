"""Prepare a disabled device-only boundary variant; no model or weight-loader integration."""
import hashlib,json,pathlib
p=pathlib.Path(__file__).resolve().parent;original=p/'pr50800_offline/device_only.cu';s=original.read_text();d=p/'pr50800_boundaries';d.mkdir(exist_ok=True);edits=[]
def change(old,new,n=1):
 global s
 assert s.count(old)==n,(old,s.count(old),n)
 s=s.replace(old,new);edits.append(dict(old=old,new=new,count=n))
change('return prmt(v, 0, sel) * 16u + 0x5A805A80u;', 'return prmt(v, 0, sel) * 16u + 0x3B003B00u;')
change('  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), h ? __high2bfloat162(sv) : __low2bfloat162(sv));', '''  const uint32_t factor_bits = 0x7E807E80u; // BF16 2^126, exactly representable
  const __nv_bfloat162 factor = *reinterpret_cast<const __nv_bfloat162*>(&factor_bits);
  const __nv_bfloat162 q = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), factor);
  __nv_bfloat162 r = __hmul2(q, h ? __high2bfloat162(sv) : __low2bfloat162(sv));''')
change('const float sl = __bfloat162float(__float2bfloat16(__fdividef(gv, 1.f + __expf(-gv))));','const float sl = __fdividef(gv, 1.f + __expf(-gv));')
change('const float sc = gscale * 9.223372036854775808e18f;','const float sc = __bfloat162float(__float2bfloat16_rn(gscale));')
for offset in ('',' + 1',' + 2',' + 3',' + 4',' + 5',' + 6',' + 7'):
 change('acc[b][4 * j'+offset+'] * sc','__bfloat162float(__float2bfloat16_rn(acc[b][4 * j'+offset+'])) * sc')
# Clear stale upstream arithmetic descriptions without changing any scheduling.
change('// BF16; the 2^63 goes on the output with the global scale. decode4 does all four A pairs of a word in 6 ALU ops and 3', '// BF16 in the upstream kernel. THIS VARIANT normalizes FP4 first and uses unshifted scales; decode4 uses6 ALU ops and3')
change('// Two block scales (bytes 0, 1 of v) -> BF16 bits of scale * 2^63, each in both halves. The pack stores each E4M3','// Two block scales (bytes0,1 of v) -> full-value BF16 scale bits, each in both halves. The pack stores each E4M3')
change('// code * 16 + (181 << 7).','// code *16 +(118 <<7) in this unshifted variant.')
change('// (named barrier 3, from tile 1 on), scale the accumulators by the global scale times 2^63 (decode4\'s exponent offset)','// (named barrier3, from tile1 on), round the FP32 accumulators to BF16, then multiply BF16 global scale')
# Record clear limits without editing the upstream source or building a loader.
s='''// DISABLED RESEARCH VARIANT. No runtime integration or H200 execution.
// Positive finite E4M3 scales only, matching upstream repacker domain.
// Upstream pack erases signs at zero scale; no claim of original-code signed-zero parity.
// Full-value coefficients; BF16 GEMM output then BF16 post-scale; no SiLU intermediate round.
// Accumulation order remains donor-specific. Raw route outputs still need Muse postnorm.
'''+s
raw=s.encode();(d/'device_only.cu').write_bytes(raw)
r=dict(parent_source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),variant_source_sha256=hashlib.sha256(raw).hexdigest(),edits=edits,not_integrated=True,device_execution=False,all_scheduling_and_route_mapping_unchanged=True,limitations=['Original upstream scale-zero pack loses coefficient zero signs.','Negative scales/negative zero/NaN are outside upstream pack domain.','GEMM FP32 order differs from Humming.','Host loader, TMA-map dispatch, global-scale owner/reload and Muse postnorm adapter absent.'])
(d/'transformation.json').write_text(json.dumps(r,indent=2))
# Exactly reversible edits after removing the5line research banner.
back='\n'.join(s.split('\n')[5:])
for e in reversed(edits):back=back.replace(e['new'],e['old'])
assert back==original.read_text()
compile_source=(p/'pr50800_offline/compile.py').read_text();compile_source=compile_source.replace("'pr50800_sm90a.cubin'","'boundary_sm90a.cubin'").replace('upstream_unmodified_math=True','boundary_variant=True')
(d/'compile.py').write_text(compile_source)
print(json.dumps({k:v for k,v in r.items() if k!='edits'},indent=2))
