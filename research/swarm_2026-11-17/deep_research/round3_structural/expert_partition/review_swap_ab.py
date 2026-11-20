from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;H=P.parent/'hopper_moe'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=0
for swap in (False,True):
 for count in range(25):
  seen=set()
  for block in range(256//(16 if swap else 8)):
   nbase=block*(16 if swap else 8)
   for t in range(3 if swap else 2):
    for lane in range(32):
     for r in range(4):
      row=t*8+(lane%4)*2+r%2 if swap else t*16+lane//4+(r//2)*8
      n=nbase+lane//4+(r//2)*8 if swap else nbase+(lane%4)*2+r%2
      if row<count:
       assert (row,n) not in seen;seen.add((row,n));checks+=1
  assert seen=={(row,n) for row in range(count) for n in range(256)}
for banks in (1,2):
 for K in (128,4096):
  seq=[[outer+64*b+16*i+kk for outer in range(0,K,64*banks) for i in range(4) for kk in range(16)]for b in range(banks)]
  assert sorted(sum(seq,[]))==list(range(K))
files=[H/f for f in ['swap_ab_probe.cuh','swap_ab_probe.py','probe_swap_ab.py','SWAP_AB_CPU_CONTRACT.json','SM86_COMPACT_E8_W13_JOBS.json']]
r={'status':'NO_DECISIVE_CPU_SOURCE_BLOCKER','source_hashes':{str(f):sha(f) for f in files},'independent_output_ownership_checks':checks,'K_bank_partition_checks':4,'primary_fragment_reference':'https://docs.nvidia.com/cuda/pdf/ptx_isa_8.5.pdf','review':['PTX A BF16 packed register order row0K0,row8K0,row0K8,row8K8 matches native m16n8k16 layout. B fragments use column=lane/4 and K=2*(lane%4)+8*r. C mapping matches standard row-major and explicit swap transpose.','Canonical WGMMA-transposed packed inverse and scale permutation agree with prior independent literal-pack oracle. Finite signed quarter-FP4*E4M3 decode including signed zero matches materializer contract.','1/2 banks each preserve K traversal with final bank0+bank1. Pinned reference BK=64*banks/WK64 is a meaningful matched control, but only GPU strict equality can validate instruction internal reduction.','Front-packed unique slots are asserted in harness; 12 generations include empty and subtile boundaries. Sentinel reads masked before activation pointer access; canonical destination bounds checked before store.','Separate FP32 swap/standard check catches differences hidden by BF16; standard-versus-pinned is mandatory. Swap failure is retained rather than converted into a tolerance pass.'],'hardening_requested':['unpack method must check input dtype/shape/device/current device and byte interval alias before broader reuse','pin config manifest and CPU contract in runtime report; assert distinct output pointers'],'limits':'CPU review only. Signed finite operands supported; NaN payload fidelity excluded. E1 fixed N256/K128 or4096, 24 rows. No performance, serving, native Hopper or pipeline qualification.'}
(P/'SWAP_AB_INDEPENDENT_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n');print(checks)
