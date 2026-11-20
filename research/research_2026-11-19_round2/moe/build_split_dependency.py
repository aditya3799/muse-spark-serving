"""Distinct scheduling variant with a consumed decoded-value wait operand."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
parent=here/'hopper_expert_split_ready.cu';s=parent.read_text()
helper=r'''// The hint is consumed by TRYWAIT: every decoded pair must exist before polling.
// It changes only the optional suspension hint, never address, parity or math.
__device__ __forceinline__ void wgs_wait_decoded(
    uint64_t* bar, uint32_t parity, const uint32_t (&a)[2][4]) {
  const uint32_t bits = a[0][0] ^ a[0][1] ^ a[0][2] ^ a[0][3]
                      ^ a[1][0] ^ a[1][1] ^ a[1][2] ^ a[1][3];
  const uint32_t hint = 128u | (bits & 127u);
  asm volatile(
      "{\n .reg .pred p;\n DECODED_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
      "@!p bra DECODED_WAIT_%=;\n}\n"
      :: "r"(smem_u32(bar)), "r"(parity), "r"(hint) : "memory");
}

'''
needle='template <int NTT> struct WGSMMA;'
assert s.count(needle)==1;s=s.replace(needle,helper+needle)
old='''if (ci == 0) wgs_wait(S.bars + BX_F + xs,
              (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);'''
new='''if (ci == 0) wgs_wait_decoded(S.bars + BX_F + xs,
              (uint32_t)(seq / WGS_NX) & 1u, af[0]);'''
assert s.count(old)==1;s=s.replace(old,new)
assert s.replace(new,old).replace(helper,'')==parent.read_text()
p=here/'hopper_expert_split_dependency.cu';p.write_text(s)
r=dict(parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_bytes=p.stat().st_size,changes=['one new acquire wait helper','activation wait calls helper with all8 first-group decoded pair registers'],hint_range_ns=[128,255],semantic_scope='Barrier address/parity/acquire and all arithmetic unchanged. Optional suspension hint is now data-dependent; runtime polling cost and progress need H200 validation.',isa='https://docs.nvidia.com/cuda/pdf/ptx_isa_8.7.pdf',no_gpu_execution=True)
(here/'split_dependency_source_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
c=(here/'compile_wgs_split_ready.py').read_text(encoding='utf-8-sig').replace('hopper_expert_split_ready.cu','hopper_expert_split_dependency.cu').replace('wgs_split_ready_','wgs_split_dependency_')
(here/'compile_wgs_split_dependency.py').write_text(c)
