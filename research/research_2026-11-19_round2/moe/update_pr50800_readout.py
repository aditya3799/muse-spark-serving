from pathlib import Path
p=Path('serving/research_2026-11-19_round2/moe/PR50800_READOUT.txt');s=p.read_text()
old='''Root's unmodified upstream hazard checker found complete CFG/zero issues for384RS
WGMMA sites (224grouped+160pair); receipt coordination/wgmma_hazard_audit/sass_audit.json.
This is one known static compiler-hazard screen, not numerical/progress/performance proof.'''
new='''The initial direct upstream hazard-checker completeness claim was SUPERSEDED:
independent review found missing BRA.DIV fallthrough edges. Root's corrected vendor-CFG
supplement binds the exact binaries and covers those edges. Explicit final receipts are
coordination/wgmma_hazard_audit/pr50800_offline/complete_cfg_audit.json and
pr50800_boundaries/complete_cfg_audit.json, each384RS sites/zero reported issues, with
missing-edge and injected-register-write controls. These are static known-hazard
screens only; independent review/status is recorded in root's final READOUT/manifest.
They do not prove hardware numerical results, progress or performance.'''
assert old in s;s=s.replace(old,new)
s+='''
Separate disabled variants qualified after the original source audit:
- Boundary-restored device source90d99adc9f217d6180e2384a0298aba7298335653cb966691a2f98b9e0d2451e
  normalizes tinyFP4 first, uses full-value SF, rounds GEMM toBF16 beforeBF16global
  multiplication, and removes upstream's extraSiLU round. Scheduling/rawroute code
  is unchanged. CUDA13.0.88 SM90a compile10.48s/184768KiB; both kernels168regs/no
  spills. Cubinf976d14146c05e9d32cd1646d2a93fd4b7395497abe6ac3a4643bb31dbef7079.
  CPU2016nonzero-positive-SF pairs exact;8zero-SF sign losses inherited upstream.
  Prefill/decode independent source reviews found no omitted arithmetic path.
- Raw-scale layout sourcef98a308049919a7df2ddc732f049072a676e24d093140d7e56490fa5bbbd7246
  is UNCOMPILED. It keeps originalE4M3 bytes and originalFP4 codes, eliminating the
  irreversible zero/subnormal alias. Separate forward/directWGS/inverse CPU oracles
  pass100663296nibbles/6291456scale bytes, all256scale codes, unchangedweights and
  canaries, across fourE1 full-widthW13/W2 cases and two seeds. The final forward
  W13row order is independently derived by reshape/stack; the earlier shared-helper
  receipt is preserved as weaker preliminary evidence. All4064finite signedSF/code
  pairs, including32signedzero and224subnormal-scale pairs, matchCPU BF16bits.
  Rawbytes127/255 survive the layout but NaNnumerical behavior remains unqualified.
  No GPUconverter, scanner/lease integration, loader or residency hook exists.
- CPU source-equation checks cover72 changed-route/single/pair tile ownership cases
  throughM16384. An abstract FIFO progressmodel passes640 schedules with stalled
  static-owner and reversed-dependency negative controls. Neither executes CUDA or
  proves TMA/mbarrier/cluster/PDL visibility/progress.

The raw layout must NEVER be passed to the normalized-scale donor kernels. The
three variants have distinct storage/math contracts and are not interchangeable.
Future prerequisites are in PR50800_H200_PLAN.txt. No2x evidence or submission.
'''
p.write_text(s)
