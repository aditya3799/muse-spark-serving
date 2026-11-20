"""Standard-library symbolic model of pinned smem_reducer.cuh; no torch/CUDA."""
from pathlib import Path
import json,hashlib,struct,math
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
INC=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
H=P.parent/'hopper_moe'
def rn(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def add(a,b):return rn(a+b)
def symbolic_add(a,b):return ('add_rn32',a,b)
def align(n,a=1024):return (n+a-1)//a*a
cases=[];assertions=0
for kind,m,bn,wn,stages in [('native_w2',8,128,32,4),('native_w2',24,128,32,4)]+[('future_swap',m,256,64,3)for m in (8,16,24)]:
 groups=bn//wn;assert groups==4
 scalars=m*wn//32
 num_values=scalars//4;num_iters=math.ceil(m/16);per=math.ceil(num_values/num_iters)
 regs={(b,g,l,s):('p',b,g,l,s)for b in range(4)for g in range(groups)for l in range(32)for s in range(scalars)}
 maxbyte=-1;barriers=0;phasechecks=0
 for chunk in range(num_iters):
  for step in (1,2):
   shared={};writes={}
   for bank in range(4):
    if bank>=4//step:continue
    if bank<4//(2*step):continue
    buffer=bank%(4//(2*step))
    for group in range(groups):
     for lane in range(32):
      warp=4*bank+group
      assert warp//4==bank and warp%4==group
      for v in range(per):
       value=chunk*per+v
       if value>=num_values:continue
       for component in range(4):
        scalar=value*4+component
        offset=((((buffer*groups+group)*per+v)*32+lane)*16+component*4)
        assert offset not in shared
        shared[offset]=regs[bank,group,lane,scalar];writes[offset]=(bank,group,lane,scalar)
        maxbyte=max(maxbyte,offset+3)
   barriers+=1 # every math thread participates after write/before read
   for bank in range(4//(2*step)):
    buffer=bank%(4//(2*step))
    for group in range(groups):
     for lane in range(32):
      for v in range(per):
       value=chunk*per+v
       if value>=num_values:continue
       for component in range(4):
        scalar=value*4+component
        offset=((((buffer*groups+group)*per+v)*32+lane)*16+component*4)
        expected_bank=bank+4//(2*step)
        assert writes[offset]==(expected_bank,group,lane,scalar)
        regs[bank,group,lane,scalar]=symbolic_add(regs[bank,group,lane,scalar],shared[offset])
        phasechecks+=1
   barriers+=1 # read finished before shared address reuse
 for g in range(groups):
  for l in range(32):
   for s in range(scalars):
    p=lambda b:('p',b,g,l,s)
    assert regs[0,g,l,s]==symbolic_add(symbolic_add(p(0),p(2)),symbolic_add(p(1),p(3)))
    assertions+=1
 actual=2*groups*per*32*16
 allocated=max(16*bn*32//128*2,m*bn//8)*16
 assert maxbyte<actual<=allocated
 stage=align(m*256*2+bn*256//2+(256//16)*bn)
 smem=align(max(stage*stages,align(allocated))+8*m)
 cases.append(dict(kind=kind,BM=m,BN=bn,WN=wn,BK=256,WK=64,threads=512,stages=stages,scalars_per_lane=scalars,values=num_values,chunks=num_iters,values_per_chunk=per,shared_reducer_bytes=actual,allocated_reduce_bytes=allocated,max_reducer_byte=maxbyte,total_shared_bytes=smem,barriers_per_thread=barriers,verified_scalar_reads=phasechecks,SM86_99KiB_fit=smem<=99*1024,SM90_227KiB_fit=smem<=227*1024))
fixtures=[]
for xs in ([2**24,1,-2**24,1],[2**24,-2**24,1,1]):
 p=list(map(rn,xs));good=add(add(p[0],p[2]),add(p[1],p[3]));adj=add(add(p[0],p[1]),add(p[2],p[3]));seq=add(add(add(p[0],p[1]),p[2]),p[3]);cross=add(add(p[0],p[3]),add(p[1],p[2]))
 assert good!=adj and good!=seq
 fixtures.append(dict(partials=p,pinned=good,wrong_adjacent=adj,wrong_left_fold=seq,other_cross_pair=cross,pinned_hex=struct.pack('<f',good).hex()))
# K identity proof for4096: stage s, bank b, inner16-slab j.
k_schedule={b:[256*s+64*b+16*j for s in range(16)for j in range(4)]for b in range(4)}
assert sorted(k for starts in k_schedule.values()for base in starts for k in range(base,base+16))==list(range(4096))
assert all(v==sorted(v)for v in k_schedule.values())
sources=[INC/'epilogue/smem_reducer.cuh',INC/'utils/context.cuh',INC/'utils/storage.cuh',INC/'mma/wgmma.cuh',INC/'epilogue/smem_writer.cuh',H/'SM90_COMPACT_JOBS.json',Path(__file__)]
r=dict(status='CPU_SYMBOLIC_FOUR_BANK_CONTRACT_PASS_NO_RUNTIME',tree='RN32(RN32(p0+p2)+RN32(p1+p3))',cases=cases,final_scalar_tree_assertions=assertions,cancellation_negative_controls=fixtures,k16_slab_starts=k_schedule,source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in sources},lane_contract={'warp':'4*bank+group for native BN128/WN32 and proposed BN256/WN64, BM=WM','group':'warp %4; bank=warp//4; lane=threadIdx.x%32','register':'Flattened C scalar s=4*(chunk*values_per_chunk+value)+component. No lane shuffle or fragment transpose in reducer. Compare corresponding output identities after each primitive mapping, never assume WGMMA and MMA flattened C indices coincide.','shared_byte':'((((buffer*4+group)*values_per_chunk+value)*32+lane)*16+component*4)','phase1':'i=1: bank2 writesbuffer0,bank3 writesbuffer1; bank0+=bank2 and bank1+=bank3. All512 maththreads synchronize twice.','phase2':'i=2: bank1 writesbuffer0; bank0+=bank1. Banks2/3 only synchronize. All512 maththreads synchronize twice.','result_writer':'Only first num_math_threads/K_WARPS=128threads write final output; other bank registers need not contain final sum.','chunking':'Two iterations for BM24; boundaries partition fragment storage, not arithmetic. Shared buffers reused only after second barrier.'},microprobe_contract=['First reducer-only probe injects independently labeled FP32 bank/lane/fragment partials and both cancellation fixtures; capture changed data, poison output, compare raw FP32 bits before BF16 conversion. Distinct lanes/groups/scalars must carry distinct labels to detect cross-lane/index mixing.','Negative controls must reproduce adjacent-pair and left-fold mismatch; a single positive random test is insufficient. CPU fixed constants are reducer fixtures, not a proof that actual GEMM generates those partials.','Then four-bank native arithmetic probe preserves per-bank K traversal: stage256*s, bank64*b, inner slab16*j; no reassignment of K segments, no two-bank emulation, no atomics, no reduced precision. Compare raw per-bank accumulators by canonical output coordinate before final reduction.','Check final raw FP32, BF16 conversion, pinned BF16 times4 boundary, global postscale; changed routes/input/weights/scales, invalid sentinels and retained output. Native WGMMA versus swap MMA mathematical equivalence still needs runtime evidence.','Four-bank swap BN256 is512threads and >=123904B shared, not SM86-admissible under99KiB. Keep current two-bank gate unchanged. SM90 offline extension must explicitly admit BK256 and W2N4096 only after source/mapping checks, preserve E restrictions.','Compile reports must include threadcount, shared bytes, registers, spills, stack and exact flags. At512threads and65536registers/SM, at most128 allocated32bitregisters/thread can fit one CTA before other limits; source declared arrays do not predict occupancy.','No duplicate persistent weights: use canonical packed NVFP4/scales; fallback uses same representation. No GPU/compiler or H200 execution performed by this contract.'])
(P/'FOUR_BANK_REDUCER_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(status=r['status'],scalar_assertions=assertions,phase_reads=sum(c['verified_scalar_reads']for c in cases),fixtures=fixtures,resource_cases=cases)))
