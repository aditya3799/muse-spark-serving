"""Independent route ownership oracle for PR50800 source equations; CPU only."""
import pathlib,json,hashlib,numpy as np
P=pathlib.Path(__file__).resolve().parent;source=P/'sources/nov10_sweep/pr50800/nvfp4_w4a16_sm90.cuh'
E=32;TOP=8;rows=[]
for seed in (930401,930409):
 rng=np.random.default_rng(seed)
 for M in (8,128,256,2048,4097,16384):
  R=M*TOP
  for kind in ('uniform','all_local_hot','all_remote'):
   ids=rng.integers(-4,256,R,dtype=np.int32)
   if kind=='all_local_hot':ids[:]=3;ids[::17]=7
   if kind=='all_remote':ids[:]=-1
   valid=(ids>=0)&(ids<E);counts=np.bincount(ids[valid],minlength=E)
   # Independent scatter oracle: arbitrate local routes randomly, but every local expert has a contiguous segment.
   segments=[];start=np.zeros(E,np.int64);cursor=0
   for e in range(E):
    rr=np.flatnonzero(ids==e);rng.shuffle(rr);start[e]=cursor;cursor+=len(rr);segments.append(rr)
   remote=np.flatnonzero(~valid);rng.shuffle(remote);order=np.concatenate(segments+[remote]);assert len(np.unique(order))==R
   perm=np.lexsort((np.arange(E),-counts));n=counts[perm];ntt=(n+127)//128
   tsz=np.where(ntt>0,((n+np.maximum(ntt,1)-1)//np.maximum(ntt,1)+31)//32*32,0)
   assert int(ntt.sum())<=E+R//128
   meta_bound=9*E+68+R//128;cnt_end=8*E+4+int(ntt.sum());assert cnt_end<=meta_bound
   for pair in (False,True):
    alljobs=0
    for rbcount in (32,16):
     offsets=np.concatenate(([0],np.cumsum(ntt*rbcount)));owner=np.zeros((R,rbcount),np.uint8)
     for tile in range(int(offsets[-1])):
      pos=int(np.searchsorted(offsets,tile,side='right')-1);lt=tile-int(offsets[pos]);tiles=int(ntt[pos]);e=int(perm[pos])
      if pair:
       crank=tile&1;qr=(lt>>1)//tiles;rb=2*qr+crank;tt=(lt>>1)-qr*tiles
      else:rb=lt//tiles;tt=lt-rb*tiles
      row0=int(start[e]+tt*tsz[pos]);nv=min(int(tsz[pos]),int(counts[e]-tt*tsz[pos]));assert 0<nv<=128
      indices=order[row0:row0+nv];assert np.all(ids[indices]==e);owner[indices,rb]+=1;alljobs+=1
     assert np.all(owner[valid]==1) and np.all(owner[~valid]==0)
    out=np.zeros(R,np.int64);local_order=order[:cursor];out[local_order]=local_order*257+ids[local_order]+1
    expected=np.where(valid,np.arange(R)*257+ids+1,0);assert np.array_equal(out,expected)
    rows.append(dict(seed=seed,M=M,profile=kind,pair=pair,local_routes=int(cursor),jobs=alljobs,metadata_end=cnt_end,metadata_capacity=meta_bound,ownership_exact=True))
r=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=len(rows),passed=True,rows=rows,scope='CPU source-equation oracle, not execution of the CUDA kernels or proof of atomic/barrier visibility.',negative_cases=['Out-of-range IDs of both signs mapped to remote zero slots.','All-remote routing yields zero GEMM jobs.'],not_tested=['Hopper runtime numerical results','weight layout conversion','GPU race/progress','CUDA graphs','allocation/serving peak','timing'])
(P/'pr50800_route_ownership.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='rows'})
