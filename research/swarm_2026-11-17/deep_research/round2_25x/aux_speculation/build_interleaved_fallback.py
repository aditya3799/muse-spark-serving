from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;S=P.parents[3]/'sglang/python/sglang/kernels/jit/csrc/elementwise/activation.cuh'
s=S.read_text();old="""  const auto gate = device::load_as<vec_t>(params.input, input_offset);
  const auto up = device::load_as<vec_t>(params.input, input_offset + num_vecs);"""
new="""  static_assert(!kReuseInput, "Interleaved fallback is out-of-place only");
  const auto pair_offset = token_id * (num_vecs * 2) + 2 * offset;
  const auto pairs_lo = device::load_as<vec_t>(params.input, pair_offset);
  const auto pairs_hi = device::load_as<vec_t>(params.input, pair_offset + 1);
  vec_t gate, up;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const int gi = 2 * i;
    const int ui = gi + 1;
    gate[i] = gi < kVecSize ? pairs_lo[gi] : pairs_hi[gi - kVecSize];
    up[i] = ui < kVecSize ? pairs_lo[ui] : pairs_hi[ui - kVecSize];
  }"""
assert s.count(old)==1
out=s.replace(old,new).replace('ActivationKernel','InterleavedActivationKernel')
(P/'interleaved_activation.cuh').write_text(out)
checks=0
for vector in (4,8,16):
 for hidden in (16,64,128,512,4096):
  for rows in (1,7,8,17,32):
   read=set()
   for row in range(rows):
    for offset in range(hidden//vector):
     pairbase=row*2*hidden+offset*vector*2
     for i in range(vector):
      gi=pairbase+2*i;ui=gi+1
      assert gi//(2*hidden)==row and ui//(2*hidden)==row
      assert (gi-row*2*hidden)//2==offset*vector+i
      assert (ui-row*2*hidden)//2==offset*vector+i
      assert gi not in read and ui not in read
      read.update((gi,ui));checks+=1
   assert len(read)==rows*hidden*2
# Reverse substitution demonstrates arithmetic, rounding, expert filtering and stores unchanged.
assert out.replace(new,old).replace('InterleavedActivationKernel','ActivationKernel')==s
report={'status':'CPU_INDEX_AND_SOURCE_CONTRACT_PASS_GPU_NOT_RUN','pair_checks':checks,'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(out.encode()).hexdigest(),'arith_unchanged':True,'scope':'Only two vector input loads become two adjacent-vector loads plus register lane permutation. Native activation arithmetic, casts, filtered invalid-row return and output store unchanged. New wrapper must expose filtered out-of-place API only.','limits':['No CUDA compile/numerical execution yet.','Does not itself implement interleaved weight packing or tagged full-helper dispatch.','Canonical checkpoint layout is preserved on unsupported architecture/weight geometry at preparation.','Runtime unsupported fusion geometry on an interleaved layer must use this activation adapter, never ordinary split-half activation.']}
(P/'FALLBACK_CPU_CONTRACT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'pairs':checks}))
