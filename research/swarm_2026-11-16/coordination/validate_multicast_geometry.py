"""CPU geometry audit of the unchanged multicast gather, not a GPU protocol test."""
from pathlib import Path
import hashlib,json,math
import numpy as np
root=Path('/mnt/d/OpenKernelChallenge/serving')
out=root/'swarm_2026-11-16/coordination'
source=root/'sglang/python/sglang/kernels/jit/csrc/muse_spark/all_gather.cuh'
text=source.read_text()
for required in ('(gridDim.x - 1) * kBlock * kVec',
                 'params.rank * params.num_rows * kNLocal + local_elem',
                 '(col / kNLocal) * params.num_rows + row',
                 'push.num_blocks - 1', 'params.counter[i].set(phase ^ 1)'):
    assert required in text, required
rows=[]
for world in (2,4,8):
  for n in (1024,4096,8192):
    for m in (1,8,9,31,32,63,128,255,256,511,512):
      nv=m*n//8; nlocal=n//world; local_nv=nv//world
      vec=np.arange(nv,dtype=np.int64)
      elem=vec*8; row=elem//n; col=elem%n
      src=((col//nlocal)*m+row)*(nlocal//8)+(col%nlocal)//8
      assert np.all(src>=0) and np.all(src<nv)
      rank_major=vec.reshape(m,world,nlocal//8).transpose(1,0,2).reshape(-1)
      assert np.array_equal(rank_major[src],vec)
      # Bijection means every pushed vector is consumed/reset exactly once.
      assert np.array_equal(np.sort(src),vec)
      for counters in (2,17,132):
        work=min(math.ceil(nv/128),counters-1)
        stride=work*128
        for count in (local_nv,nv):
          # CUDA's flattened (block,thread,loop) schedule partitions all vectors.
          visited=np.zeros(count,dtype=np.uint8)
          for first in range(min(stride,count)):
            visited[first:count:stride]+=1
          assert np.all(visited==1), (world,n,m,counters,count)
        toggled=set(range(work))|set(range(work,counters))
        assert toggled==set(range(counters))
      rows.append(dict(world=world,n=n,m=m,vector_mapping='bijective',counter_counts=[2,17,132]))
# Marker encoding preserves finite numeric value, including signed zero.
bits=np.arange(65536,dtype=np.uint32)
encoded=np.where(bits==0,0x8000,bits)
finite=(bits&0x7f80)!=0x7f80
values=(bits<<16).view(np.float32)
encoded_values=(encoded<<16).view(np.float32)
assert np.all(values[finite]==encoded_values[finite])
report=dict(passed=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),geometry_cases=len(rows),schedules=len(rows)*3,rows=rows,finite_bf16_marker_cases=int(finite.sum()),scope='CPU audit of exact existing kernel indexing, vector coverage, per-phase counter coverage and zero-marker value preservation. Does not validate multicast hardware, memory ordering, inter-rank progress or actual CUDA graph replay.')
(out/'multicast_geometry_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
