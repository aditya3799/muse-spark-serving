import json,hashlib,math
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[3]
INC=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
H=P.parent/'hopper_moe'
def align(n,a=1024): return (n+a-1)//a*a
cases=[]
for m in (8,16,24):
 for bk in (64,128):
  banks=bk//64; threads=128*banks
  stage=align(m*bk*2+256*bk//2+(bk//16)*256)
  allocated_reduce=max(16*256*32//128*(banks//2),m*256//8)*16
  c_scalars=4*(m//8)*4
  values=c_scalars//4; iters=math.ceil(m/16); per=math.ceil(values/iters)
  actual_reduce=(banks//2)*4*per*32*16
  assert actual_reduce<=allocated_reduce
  reached=[]
  for t in range(iters):
   for i in range(per):
    if t*per+i<values: reached.append(t*per+i)
  assert reached==list(range(values))
  # Unchanged tree: each scalar in bank0 accumulates bank1 exactly once for two banks.
  writes=set()
  for nw in range(4):
   for lane in range(32):
    sub=(lane%4)*2+(lane%8)//4
    for i in range(4):
     for j in range(m//8):
      for p in range(2):
       row=8*j+sub; col=((2*i+p)^sub)*4+lane//8
       idx=row*32+col+nw*m*32
       assert 0<=idx<m*256//2
       assert idx not in writes
       writes.add(idx)
  assert len(writes)==m*256//2
  total=align(max(stage*3,align(allocated_reduce))+8*m)
  cases.append(dict(BM=m,WM=m,BN=256,WN=64,BK=bk,WK=64,K_banks=banks,threads=threads,stage_bytes=stage,shared_bytes=total,declared_C_scalars=c_scalars,declared_register_array_words=40+m//2+c_scalars,actual_reducer_bytes=actual_reduce,allocated_reducer_bytes=allocated_reduce,reducer_value_coverage=len(reached),writer_half2_coverage=len(writes),SM86_99KiB_fit=total<=99*1024,SM90_227KiB_fit=total<=227*1024))
sources=[H/'swap_ab_humming.cuh',H/'swap_ab_humming_codegen.py',INC/'utils/storage.cuh',INC/'epilogue/smem_reducer.cuh',INC/'epilogue/smem_writer.cuh',H/'SM90_COMPACT_JOBS.json']
orig=json.loads((H/'SM90_COMPACT_JOBS.json').read_text(encoding='utf-8-sig'))
native=[dict(job=j['job_id'],block=j['baseline']['block_shape'],warp=j['baseline']['warp_shape'],mma=j['baseline']['mma_type'],stages=j['baseline']['num_stages'],banks=j['baseline']['block_shape'][2]//j['baseline']['warp_shape'][2]) for j in orig['jobs']]
r=dict(status='CPU_RESOURCE_AND_INDEX_CAPACITY_PASS_NOT_COMPILED',cases=cases,native_SM90_original_selectors=native,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},limitations=['Pure standard-library model; no CUDA context, compiler, or runtime. Shared bytes derived literally from pinned storage.cuh with indexed metadata and three stages. C++ sizeof and PTXAS resources must confirm.','Declared array words are source payload capacity, not allocated registers or live-register lower bound; compiler lifetime reuse, temporaries and arithmetic arrays may change pressure. No occupancy assertion.','Writer model proves unique in-range half2 destinations and reducer scalar coverage, not data values or activation-load correctness. Expert owns independent mapping review.','SM90 W2 light uses four BK256/WK64 banks outside current guard and one/two-bank numerical microgate. Do not silently substitute BK128.','SM90 W13 uses WGMMA and WN32, while candidate uses MMA and WN64. Equal K-bank count is necessary but not a proof of native WGMMA arithmetic equivalence.','Existing guard limits N256/8192, K128/4096 and E1/8: native W2 N4096 and scored E32 are rejected. This is a W13 research adapter only until explicit qualification.','No additional persistent weight representation: canonical packed weight/scale addressing must remain directly consumed, with fallback using same original representation.','Initial runtime should use exact original configuration comparator and matched native-MMA comparator, changed routes/weights/scales, poisoned inactive slots, duplicate controls, then full helper if stage passes.'],ranked_plan=['First existing BM16 BK128 compiled gate with fixed128/256-thread contract and compiler-reported shared bytes/registers/spills.','Then BM8 and BM24 only after writer/reducer/index proofs and unchanged K-bank arithmetic gates; retain source-specific guard rejections.','SM90 offline source/resource gate may test this MMA alternative, but does not establish WGMMA equivalence or four-bank W2 support. No H200 execution authorized.'])
(P/'SWAP_AB_RESOURCE_ADMISSION_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'status':r['status'],'cases':len(cases),'writer_half2_checks':sum(c['writer_half2_coverage'] for c in cases),'resources':[(c['BM'],c['BK'],c['shared_bytes']) for c in cases]}))
