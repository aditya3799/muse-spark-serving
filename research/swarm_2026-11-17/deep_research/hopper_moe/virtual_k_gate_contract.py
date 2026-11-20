"""CPU-only gate contracts; importing does not initialize CUDA."""
import random

def require_h200(major,minor,name):
 if (major,minor)!=(9,0) or 'H200' not in name.upper():raise RuntimeError('This gate requires an NVIDIA H200 SM90; no SM86 execution or performance proxy is permitted')

def route_arrays(m,experts,block,pattern,seed):
 if m<=0 or experts<8 or block<=0:raise ValueError('Invalid geometry')
 rng=random.Random(seed);slots=m*8
 if pattern=='uniform':ids=[rng.randrange(experts*8) for _ in range(slots)];ids=[x if x<experts else -1 for x in ids]
 elif pattern=='hot_expert':ids=[0]*slots
 elif pattern=='mixed_duplicate':ids=[0 if i%16<8 else -1 for i in range(slots)]
 elif pattern=='all_remote':ids=[-1]*slots
 else:raise ValueError(pattern)
 capacity=((slots+experts*(block-1)+block-1)//block)*block
 sorted_ids=[slots]*capacity;expert_ids=[-1]*(capacity//block);offset=0
 for e in range(experts):
  indices=[i for i,x in enumerate(ids) if x==e]
  padded=((len(indices)+block-1)//block)*block
  sorted_ids[offset:offset+len(indices)]=indices
  expert_ids[offset//block:(offset+padded)//block]=[e]*(padded//block)
  offset+=padded
 assert offset<=capacity
 return {'sorted_ids':sorted_ids,'expert_ids':expert_ids,'padded':offset,'active':[x>=0 for x in ids],'local_ids':ids}
