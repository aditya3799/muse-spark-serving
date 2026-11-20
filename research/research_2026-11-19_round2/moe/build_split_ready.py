"""Source-only split-readiness prototype; no arithmetic or ring-release change."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
parent=here/'finite_frozen/hopper_expert_register_finite.cu';s=parent.read_text()
changes=[]
def replace(old,new):
 global s
 assert s.count(old)==1,(old[:80],s.count(old));s=s.replace(old,new);changes.append((old,new))
replace('#ifndef REGISTER_FINITE','#ifndef SPLIT_READY\n#define SPLIT_READY 0\n#endif\n#ifndef REGISTER_FINITE')
replace('uint64_t* bar = S.bars + BW_F + xs;', 'uint64_t* bar = S.bars + (SPLIT_READY ? BX_F : BW_F) + xs;')
replace('k < BW_E ? 2 : k < BX_F ? NW', 'k < BW_E ? (SPLIT_READY ? 1 : 2) : k < BX_F ? NW')
needle='''        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {'''
replace(needle,'''        // Weight-ready permits register dequantization to start first.
        // Activation-ready is still acquired before the first WGMMA read.
        if constexpr (SPLIT_READY) {
          if (ci == 0) wgs_wait(S.bars + BX_F + xs,
              (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
        }
'''+needle)
replace('// Both producer arrivals and both byte counts must complete this phase.',
        '// Joined mode completes both transactions; split mode completes activation bytes here.')
replace('// The BW_F acquire above covers both weight and activation transactions.',
        '// BW_F always acquires weights; split mode acquires activation at BX_F below.')
p=here/'hopper_expert_split_ready.cu';p.write_text(s)
recovered=s
for old,new in reversed(changes):recovered=recovered.replace(new,old)
assert recovered==parent.read_text()
report=dict(parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_bytes=p.stat().st_size,reversible_source_changes=len(changes),unchanged=['coefficient arithmetic','WGMMA order','all epilogue math/global scale placement','BW_E weights release','BX_E activation release','ringdepth3','producer/consumer stage order','cooperative launch geometry'],new_contract=['BW_F count1: weights TMA only','BX_F count1: activation TMA only','consumer first register-decode group afterBW_F beforeBX_F','consumer first WGMMA only afterBX_F'],not_compiled=True,no_gpu_execution=True,source_inspiration='https://humansand.ai/blog/faster-than-light-nvfp4-decode',scope='New separation of already allocated ready barriers; Blackwell article supplies hypothesis, no Hopper speed claim.')
(here/'split_ready_source_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

