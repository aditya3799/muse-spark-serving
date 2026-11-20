"""CPU route-layout proof for pinned DeepGEMM; no GPU performance claim."""
import json,pathlib,random,hashlib
P=pathlib.Path(__file__).resolve().parent
A=128

def align(n): return (n+A-1)//A*A

def build(ids,E):
    counts=[ids.count(e) for e in range(E)]
    starts=[];ends=[];slots=[];labels=[];last=0
    for e,n in enumerate(counts):
        start=align(last);starts.append(start);last=start+n;ends.append(last)
        slots.extend([-1]*(start-len(slots)))
        labels.extend([-1]*(start-len(labels)))
        selected=[i for i,v in enumerate(ids) if v==e]
        slots.extend(selected);labels.extend([e]*n)
    capacity=align(len(ids)+E*(A-1))
    slots.extend([-1]*(capacity-len(slots)));labels.extend([-1]*(capacity-len(labels)))
    return counts,starts,ends,slots,labels,capacity

def main():
    rng=random.Random(281);checks=0;cases=0
    for E in [1,2,8,32]:
      for S in [0,1,8,31,32,127,128,129,1024,3012,16384,131072]:
        for pat in ['remote','first','last','uniform','skew']:
          if pat=='remote':ids=[-1]*S
          elif pat=='first':ids=[0]*S
          elif pat=='last':ids=[E-1]*S
          elif pat=='uniform':ids=[rng.randrange(-1,E) for _ in range(S)]
          else:ids=[0 if rng.random()<.7 else rng.randrange(-1,E) for _ in range(S)]
          counts,starts,ends,slots,labels,cap=build(ids,E)
          assert cap>=len(slots) and cap%128==0;checks+=2
          observed=[];last=0
          for e in range(E):
            # Literal psum scheduler contract: next expert begins at align(previous end).
            start=align(last);n=ends[e]-start
            assert n>=0 and n==counts[e] and start==starts[e];checks+=3
            for block in range((n+A-1)//A):
              first=start+block*A
              assert labels[first]==e;checks+=1
              for row in range(first,min(first+A,cap)):
                if row<ends[e]:
                  assert slots[row]>=0 and ids[slots[row]]==e;observed.append(slots[row]);checks+=2
                else:
                  assert slots[row]==-1 and labels[row]==-1;checks+=2
            last=ends[e]
          assert sorted(observed)==[i for i,v in enumerate(ids) if 0<=v<E];checks+=1
          assert len(observed)==len(set(observed));checks+=1
          # Padding overhead is bounded independently of token capacity.
          assert align(last)-sum(counts)<=E*(A-1)+A-1;checks+=1
          cases+=1
    negative={'counts':[1,1],'naive_ends':[1,2],'second_start':128,'second_signed_count':2-128,'second_uint32_count':(2-128)%(1<<32)}
    assert negative['second_signed_count']<0
    paths=[P/'deepgemm_pinned/tag_source/csrc/apis/gemm.hpp',P/'deepgemm_pinned/tag_source/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp',P/'deepgemm_pinned/tag_source/csrc/jit_kernels/heuristics/sm90.hpp',P/'deepgemm_pinned/tag_source/csrc/jit_kernels/heuristics/runtime.hpp',P/'deepgemm_pinned/source/deep_gemm/include/deep_gemm/scheduler/gemm.cuh']
    result={'status':'CPU_ADAPTER_CONTRACT_PASS_NOT_GPU_VALIDATION','cases':cases,'assertions':checks,'default_SM90_alignment':128,'negative_control':negative,'hashes':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'contract':['Pinned wheel0.2.3.post2 has SM90 BF16 grouped kernel and Python API; matching tag API dispatches arch9.','Psum metadata has E ends, not Humming E+1 offsets. Each expert start is align(previous end,128).','Hopper grouped tileM equals runtime alignment (default128); do not mutate global alignment silently.','Gather must zero padding, scatter only inverse valid slots. Global expert scale remains outside matrix multiplication.','For bounded expert bands, IDs must be remapped to band-local weight indices and each original slot scattered exactly once.','Dynamic GPU counts need no host synchronization; fixed-capacity buffers and GPU end metadata required.','Changing WGMMA geometry or library changes accumulation; exact dequant alone does not prove numerical/serving equivalence.'],'memory':{'extra_scratch_bound_E32_H4096_bytes':align(32*127)*3*4096*2,'W13_staging_band8_bytes':8*8192*4096*2,'note':'Bounds exclude allocator/graph/startup overhead; shared workspace across layers mandatory.'}}
    (P/'DEEPGEMM_PINNED_LAYOUT_PROOF.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':cases,'assertions':checks,'negative_control':negative}))
if __name__=='__main__':main()
