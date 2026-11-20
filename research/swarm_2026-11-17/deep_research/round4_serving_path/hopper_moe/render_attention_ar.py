"""Generate an isolated attention AR+Muse-boundary experiment, never install it."""
import hashlib,json,pathlib,re
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[3]
# ROOT is serving/swarm_2026-11-17; resolve explicitly for both Windows and WSL.
ROOT=next(p for p in HERE.parents if (p/'RULES.md').exists())
REL='sglang/python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh'
def digest(x):return hashlib.sha256(x).hexdigest()
def render():
 ledger=json.loads((HERE/'SOURCE_COLLECTIVE_AUDIT.json').read_text(encoding='utf-8-sig'))
 src=(ROOT/REL).read_bytes();assert digest(src)==ledger['source_hashes'][REL], 'source drift'
 text=src.decode().replace('\r\n','\n')
 start=text.index('template <\n    int64_t kHidden,')
 stop=text.index('// Multi-CTA path partitions each row',start)
 original=text[start:stop]
 prefix='''
  static_assert(kHidden == 8192 && kNumThreads == 256 && kWorld == 8);
  static_assert(!kUsePDL, "initial protocol experiment deliberately disables PDL");
  const auto epoch = distributed::PushEpoch<kWorld>{collective.ws};
  using Lamport = distributed::LamportTrait<bf16_t, 8, 4>;
  const uint32_t rank = collective.rank;
  const uint32_t row_vec_base = blockIdx.x * (kHidden / 8);
  // Every row's producer runs before its own consumer; peers may arrive in any order.
  for (int p = 0; p < kPacks; ++p) {
    const uint32_t idx = threadIdx.x + p * kNumThreads;
    BF16Vec local;
    local.load(branch, idx);
    Lamport::clear_pos_zero(local.data());
    for (uint32_t dst = 0; dst < kWorld; ++dst)
      ptx::st_relaxed_16B(local, epoch.slot_ptr(dst, rank), row_vec_base + idx);
  }
  BF16Vec zero;
  Lamport::fill_pos_zero(zero.data());
'''
 recv='''    BF16Vec peers[kWorld];
    bool pending;
    do {
      pending = false;
      for (uint32_t src = 0; src < kWorld; ++src) {
        ptx::ld_relaxed_16B(peers[src], epoch.slot_ptr(rank, src), row_vec_base + idx);
        pending |= Lamport::has_pos_zero(peers[src].data());
      }
    } while (pending);
    branch_v = reduce_vec(peers);  // identical rank-order FP32 sum and BF16 boundary
    for (uint32_t src = 0; src < kWorld; ++src)
      ptx::st_global_16B(zero, epoch.slot_ptr(rank, src), row_vec_base + idx);'''
 end='''  __syncthreads();
  epoch.flip();
  if (blockIdx.x == 0)
    epoch.unsafe_flip_range(params.num_rows, collective.num_counters);
'''
 candidate=original.replace('    int64_t kHidden,','    uint32_t kWorld,\n    int64_t kHidden,',1)
 candidate=candidate.replace('muse_spark_norm_residual_kernel(const __grid_constant__ NormResidualParams params)', 'research_attention_ar_norm_kernel(const __grid_constant__ ResearchAttentionARParams<kWorld> collective)')
 anchor='  using namespace device;\n'
 candidate=candidate.replace(anchor,'  const auto& params = collective.norm;\n'+anchor,1)
 wait='  PDLWaitPrimary<kUsePDL>();\n'
 assert candidate.count(wait)==1
 candidate=candidate.replace(wait,wait+prefix,1)
 load='    branch_v.load(branch, idx);'
 assert candidate.count(load)==1
 candidate=candidate.replace(load,recv,1)
 trigger='  PDLTriggerSecondary<kUsePDL>();'
 candidate=candidate.replace(trigger,end+trigger,1)
 # Reverse every permitted substitution; this is stronger than a line-count claim.
 reverse=candidate.replace(end,'').replace(recv,load).replace(prefix,'').replace('  const auto& params = collective.norm;\n','',1).replace('research_attention_ar_norm_kernel(const __grid_constant__ ResearchAttentionARParams<kWorld> collective)','muse_spark_norm_residual_kernel(const __grid_constant__ NormResidualParams params)').replace('    uint32_t kWorld,\n','',1)
 assert reverse==original
 preamble='''// RESEARCH ONLY: no live SGLang hook; no multi-GPU or serving qualification.
#pragma once
#include <sgl_kernel/distributed/communicator.cuh>
#include "norm_residual.cuh"
namespace sglang {
template<uint32_t kWorld>
struct ResearchAttentionARParams {
  NormResidualParams norm;
  device::distributed::PushWorkSpace<kWorld> ws;
  uint32_t rank;
  uint32_t num_counters;
};
'''
 out=preamble+candidate+'\n} // namespace sglang\n'
 (HERE/'attention_ar_norm.cuh').write_text(out,encoding='utf-8',newline='\n') if __import__('sys').version_info>=(3,10) else (HERE/'attention_ar_norm.cuh').write_bytes(out.encode())
 report=dict(status='RENDERED_NOT_COMPILED_NOT_EXECUTED',original_source_sha256=digest(src),original_kernel_sha256=digest(original.encode()),generated_sha256=digest(out.encode()),unchanged_norm_body_after_reversal=True,added_scope='one-shot push before identical norm; rank-order reduce replaces materialized branch load',required_guards=dict(arch='SM90 production experiment only; separate SM86 virtual-rank fixture',world_size=8,H=8192,threads=256,M='1..8 inclusive',bf16_branch=True,fp32_stream=True,PDL=False,one_shot_push_selected=True,default_JIT_boundary=True,SP=False,DP=False,all_ranks_same_M=True,num_counters='>=M; all counters phase-coherent at call entry',slot_bytes='>=M*8192*2',alias='outputs cannot alias peer plane or any live input; no early stream/weight mutation'),limitations=['No host integration or native dispatcher change','No NVIDIA memory model/runtime peer qualification','No promise compiler retains exact FMA scheduling until SASS/runtime comparison','SingleCTA-per-row may reduce communication parallelism and regress latency','Signed-zero follows exact Lamport 32-bit marker behavior of baseline, not numeric-zero canonicalization'])
 (HERE/'ATTENTION_AR_RENDER_CONTRACT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':render()
