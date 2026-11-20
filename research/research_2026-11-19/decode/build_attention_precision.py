"""Generate an auditable q_len8 donor variant preserving native precision boundaries.

Generation only. Does not execute, build or benchmark upstream downloaded code.
The resulting kernel is unqualified until Hopper differential validation.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
donor=HERE/'sources/alygg77/kernels/muse_attention_decode.py'
text=donor.read_text();changes=[]
def replace(before,after,count=1):
 global text
 actual=text.count(before)
 assert actual==count,(before[:100],actual,count)
 text=text.replace(before,after);changes.append(dict(before=before,after=after,count=count))

# The native model stores BF16 attention output before its output norm/gate.
replace('float pr8(float x) {\n  return x;\n}',
 'float pr8(float x) {\n  return __bfloat162float(__float2bfloat16_rn(x));\n}')
replace('  const float* inv_freq;', '  const float* inv_freq; // original FP32 cos/sin cache, not frequencies\n  const int64_t* positions;')
replace('torch::Tensor inv_freq, torch::Tensor out,', 'torch::Tensor inv_freq, torch::Tensor positions, torch::Tensor out,',2)
replace('  p.inv_freq = inv_freq.data_ptr<float>();','  p.inv_freq = inv_freq.data_ptr<float>();\n  p.positions = positions.data_ptr<int64_t>();')
replace('if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);',
 'if (tidL < 8) { /* RoPE reads the original caller-owned cache below. */ }')
begin=text.index('      const float ang = (float)(seq + j) * S.ifreq[i];')
end=text.index('      S.sn[e] = sv;',begin)+len('      S.sn[e] = sv;')
old=text[begin:end]
replace(old,'''      const int64_t position = p.positions[b * L + j];
      S.cs[e] = p.inv_freq[position * 64 + i];
      S.sn[e] = p.inv_freq[position * 64 + 32 + i];''')

# Parent QK RMS writes BF16 before rotary; after rotary BF16 precedes FP8.
replace('for (int e = 0; e < 16; ++e) x[e] *= r;',
 'for (int e = 0; e < 16; ++e) x[e] = pr8(x[e] * r);')
replace('for (int e = 0; e < 8; ++e) x[e] *= r;',
 'for (int e = 0; e < 8; ++e) x[e] = pr8(x[e] * r);',2)
replace('      const int hk4 = hh >> 3, gg4 = hh & 7;',
 '      for (int e = 0; e < 16; ++e) x[e] = pr8(x[e]);\n      const int hk4 = hh >> 3, gg4 = hh & 7;')
replace('    // f16 query pre-scaled to log2 units, staged',
 '    for (int e = 0; e < 8; ++e) x[e] = pr8(x[e]);\n    // f16 query pre-scaled to log2 units, staged')
replace('        sc = __fdividef(1.f, kscale);',
 '        for (int e = 0; e < 8; ++e) x[e] = pr8(x[e]);\n        sc = __fdividef(1.f, kscale);')
replace('          *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);',
 '          if (row != 0) *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);')

# Keep FP32 maxima and probability sums. E4M3 is used only by the PV multiply,
# with the native FA3 Max_offset=8 scaling to preserve its subnormal range.
replace('  float lacc[4] = {0.f, 0.f, 0.f, 0.f};',
 '  float lacc[4] = {0.f, 0.f, 0.f, 0.f}; // unused extra PV columns\n  float native_sum[2] = {0.f, 0.f};')
begin=text.index('    // after a segment\'s first tile the offset test')
end=text.index('    PROF_TILE;',begin)
old=text[begin:end]
new='''    // FA3-style FP32 online softmax. cr stores maxima in raw QK units.
    float maxima[2] = {-INFINITY, -INFINITY};
    for (int nt = 0; nt < 8; ++nt) {
      maxima[0] = fmaxf(maxima[0], fmaxf(s[4 * nt], s[4 * nt + 1]));
      maxima[1] = fmaxf(maxima[1], fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
    }
    maxima[0] = quad_max(maxima[0]); maxima[1] = quad_max(maxima[1]);
    float alpha[2] = {1.f, 1.f};
    for (int h = 0; h < 2; ++h) {
      const float previous = cr[h];
      cr[h] = first ? maxima[h] : fmaxf(previous, maxima[h]);
      const float current = cr[h] == -INFINITY ? 0.f : cr[h];
      if (!first) alpha[h] = exp2f((previous - current) * ssc);
      native_sum[h] *= alpha[h];
    }
    if (!first) {
      for (int t = 0; t < 8; ++t) {
        o[0][t][0] *= alpha[0]; o[0][t][1] *= alpha[0];
        o[0][t][2] *= alpha[1]; o[0][t][3] *= alpha[1];
      }
    }
    first = false;
    float probabilities[32];
    for (int nt = 0; nt < 8; ++nt) {
      for (int h = 0; h < 2; ++h) {
        // Match softmax.h: all-masked row uses zero max_scaled, not -8.
        const float shifted_max = cr[h] == -INFINITY ? 0.f : cr[h] * ssc - 8.f;
        for (int e = 0; e < 2; ++e) {
          const int ix = 4 * nt + 2 * h + e;
          probabilities[ix] = exp2f(fmaf(s[ix], ssc, -shifted_max));
          native_sum[h] += probabilities[ix];
        }
      }
    }
    for (int kk = 0; kk < 2; ++kk) {
      const float* v = probabilities + 16 * kk;
      pa[kk][0] = e4m3x4f(v[0], v[1], v[4], v[5]);
      pa[kk][1] = e4m3x4f(v[2], v[3], v[6], v[7]);
      pa[kk][2] = e4m3x4f(v[8], v[9], v[12], v[13]);
      pa[kk][3] = e4m3x4f(v[10], v[11], v[14], v[15]);
    }
    float* native_output = &o[0][0][0];
    for (int e = 0; e < 32; ++e) reg_fence(native_output[e]);
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
    if constexpr (LSYNC || LSYNCN) { fence_proxy_async(); wgsync(wgi); }
    wgmma_fence();
    wgmma_m64n72k32_e4m3(native_output, lacc, pa[0], vdesc_c);
    wgmma_m64n72k32_e4m3(native_output, lacc, pa[1], vdesc_c + 2);
    wgmma_commit(); wgmma_wait<0>();
    for (int e = 0; e < 32; ++e) reg_fence(native_output[e]);
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
'''
replace(old,new)
replace('  m_r[0] = first ? -INFINITY : cr[0];\n  m_r[1] = first ? -INFINITY : cr[1];\n  l_r[0] = 0.25f * lacc[0];   // every lane of the quad holds the full row sum; the epilogue quad-sums\n  l_r[1] = 0.25f * lacc[2];',
 '''  m_r[0] = first ? -INFINITY : cr[0] * ssc;
  m_r[1] = first ? -INFINITY : cr[1] * ssc;
  l_r[0] = native_sum[0]; // each quad lane contributes unrounded FP32 P
  l_r[1] = native_sum[1];''')

replace('data["k_scale"], data["v_scale"], _inv_freq(cfg["rope_theta"], device), out,',
 'data["k_scale"], data["v_scale"], data["rope_cache"], data["positions"], out,')
replace('SUPPORTED_Q_LENS = (1, 8)','SUPPORTED_Q_LENS = (8,)')
replace('name=f"okc_attn_v298{tag}_90a"','name=f"okc_attn_precision_v1{tag}_90a"')
replace('    if not _supported(data["config"]) or not _fits(data):\n        return _fallback(data)',
 '    if not _supported(data["config"]) or not _fits(data):\n        raise ValueError("Unqualified shape: production adapter must dispatch original native attention")')
text='''# RESEARCH ONLY: unqualified precision-boundary restoration, no serving integration.
# Relative to donor: native BF16 boundaries, original FP32 rotary cache,
# FP32 softmax denominator and FA3-style P*256 scaling. Different tile/split
# reductions remain; Hopper differential output/cache validation is mandatory.
'''+text
ast.parse(text)
target=HERE/'precision_attention.py';target.write_text(text)
(HERE/'precision_attention_manifest.json').write_text(json.dumps(dict(
 donor_sha256=hashlib.sha256(donor.read_bytes()).hexdigest(),
 variant_sha256=hashlib.sha256(text.encode()).hexdigest(),changes=changes,
 qualified=False,gate='Source restoration only; no Hopper execution or actual output validation yet'),indent=2))
print('generated',target.name,'bytes',len(text.encode()),'changes',len(changes))
