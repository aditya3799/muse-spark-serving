from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;ROOT=P.parents[3];INC=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
def align(v):return (v+1023)//1024*1024
cases=[]
for m in (8,16,24):
 writes=set();loads=set();scale_bytes=set()
 for bank in range(4):
  for nw in range(2):
   for lane in range(32):
    for it in range(4):
     bidx=64*it+256*bank+32*nw+lane
     assert bidx not in loads;loads.add(bidx)
     bsidx=(4*bank+it)*16+8*nw+lane//4
     assert 0<=bsidx<256
     scale_bytes.update(range(bsidx*8,bsidx*8+8))
 for nw in range(2):
  for lane in range(32):
   sub=(lane%4)*2+(lane%8)//4
   for i in range(4):
    for j in range(m//8):
     for p in range(2):
      idx=(8*j+sub)*32+((2*i+p)^sub)*4+lane//8+nw*m*32
      assert 0<=idx<m*128//2 and idx not in writes;writes.add(idx)
 assert loads==set(range(1024)) and len(scale_bytes)==2048 and len(writes)==m*128//2
 actual=2*2*((m//2+(m+15)//16-1)//((m+15)//16))*32*16
 allocation=max(16*128*32//128*2,m*128//8)*16
 assert actual<=allocation
 stage=align(m*256*2+128*256//2+16*128)
 total=align(max(stage*3,align(allocation))+8*m)
 cases.append(dict(BM=m,BN=128,WM=m,WN=64,BK=256,WK=64,threads=256,shared_bytes=total,actual_reducer_bytes=actual,allocated_reduce_bytes=allocation,B_int4_loads=len(loads),BS_byte_coverage=len(scale_bytes),writer_half2_cells=len(writes),fits_SM86_99KiB=total<=99*1024))
sources=[INC/'memory/s2r_loader/loader_b.cuh',INC/'memory/s2r_loader/loader_bs.cuh',INC/'memory/g2s_loader/loader_b.cuh',INC/'utils/context.cuh',INC/'utils/storage.cuh',INC/'epilogue/smem_writer.cuh',ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/kernel/humming.py']
r=dict(status='NARROW_BN128_FOUR_BANK_SOURCE_CAPACITY_PASS_NOT_ADMITTED_OR_COMPILED',cases=cases,proof=['MMA context permits N_WARPS2; four-warp-N condition only applies when Ctx::kUseWgmma=true. Local writer orientation does not change context.','S2R B WN64 uses full int4 per lane, TRUE_N_WARPS2, K_WARPS4; address64*iter+256*bank+32*nwarp+lane covers1024 int4 exactly.','S2R group-scale layout1 uses LoadType8bytes,64/WN=1, index(4*bank+iter)*16+8*nwarp+lane/4; covers2048B with expected4lane replication.','G2S B legacy_load_2d receives1024int4 and256threads; four full transfer rounds. No requirement N_WARPS>=4 on MMA path.','Pinned shape checks permit BN128/WN64,BK256/WK64 power-of-two ratios. Existing research adapter guards and static_assert deliberately reject these until extended.','New writer capacity/model valid for two output warp groups; no assertion of actual runtime fragment values.'],proposed_guard_extension=['Separate variant only: BN128/WN64, BK256/WK64, BM=WM8/16/24, num_stages3, indexed MMA,256threads; all original format/feature restrictions remain.','Explicit W2 research N4096,K4096,E1/8 remains separate from E32 serving eligibility. No silent architecture or dtype fallback.','Require same-canonical-weight comparator retaining original four-bank assignment and exact(p0+p2)+(p1+p3)tree.','Compile and sizeof guard before runtime: shared68608/80896/93184B plus256threads. No change to current two-bank gate.'],limitations=['No CUDA/compiler invoked; nativeWGMMA vs swapMMA arithmetic stillunproven.','Same native outputtileBN128 retained; WN32?64 and stages4?3 still change scheduler/resource geometry.','Memory footprint fits nominalSM86limit butregisters/spills/occupancyunknown.','Microkernel rawtreegate doesnot validate this pipelinedloader/epilogue integration.'],source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in sources})
(P/'FOUR_BANK_NARROW_ADMISSION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['cases']))
