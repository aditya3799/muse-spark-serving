"""CPU-only direct WGS -> raw-scale50800 layout, independent forward pack oracle.
No GPU converter or model loader is installed. Coefficients/scales are copied as bits.
"""
import pathlib,json,hashlib,ast,numpy as np
from forward_oracles import wgs_forward
P=pathlib.Path(__file__).resolve().parent;D=P/'pr50800_rawscale';D.mkdir(exist_ok=True)
rep=(P/'sources/nov10_sweep/pr50800/nvfp4_sm90_repack.py').read_text()
POS=next(ast.literal_eval(n.value) for n in ast.parse(rep).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_POS' for t in n.targets))
def kperm(k):
 p=np.arange(k);r=p%64;return p//64*64+16*((r//2)%4)+4*(r//16)+2*((r//8)%2)+r%2

def canonical_row(r,up):
 if not up:return r
 i=r//256*128+r%256//128*64+r%64
 z=i%64
 return i//64*64+16*((z//2)%4)+4*(z//16)+2*((z//8)%2)+z%2+(r//64%2)*4096

def forward(q,sf,up):
 e,n,k=q.shape
 if up:
  ki=kperm(4096).reshape(32,2,64)
  rows=np.stack((ki,4096+ki),axis=2).reshape(-1)
 else:rows=np.arange(n)
 q=q[:,rows][:,:,kperm(k)];sf=sf[:,rows]
 v=q.reshape(e,n//256,16,2,8,k//64,4,2,4,2).transpose(0,1,5,2,4,8,6,9,7,3)
 nib=(v[...,0]|v[...,1]<<4).reshape(e,n//256,k//64,16,512).copy()
 words=nib.view(np.uint32);outw=np.zeros_like(words)
 for m in range(4):
  for h in range(2):
   c=(words>>(16*h+4*m))&15
   for b,pos in enumerate(POS[m][h]):outw|=((c>>b)&1)<<pos
 out=np.empty((e,n//256,k//64,16,576),np.uint8)
 out[...,:512]=outw.view(np.uint8)
 out[...,512:]=sf.reshape(e,n//256,16,2,8,k//64,4).transpose(0,1,5,2,4,6,3).reshape(e,n//256,k//64,16,64)
 return out

def source_code(words,e,n,k,N,up):
 p=n%4096//8*16+n%8+n//4096*8 if up else n
 tile=((e*(N//256)+p//256)*16+k//256)*4+p%256//64
 ix=k%256//64*512+p%64//16*128+p%8*16+k%64//16*4+k%8//2
 w=words[tile*2304+ix];r=p//8%2+k//8%2*2;h=k%2
 mag=(w>>((6+3*r+16*h)%32).astype(np.uint32))&7
 sign=((w>>((15+3*r+16*h)%32).astype(np.uint32))&1)<<3
 sp=((w>>(4+16*h).astype(np.uint32))&1)|(((w>>(16*h).astype(np.uint32))&3)<<1)|((w>>(16*h).astype(np.uint32))&8)
 return np.where(r==3,sp,mag|sign)

def source_scale(raw,e,n,g,N,up):
 p=n%4096//8*16+n%8+n//4096*8 if up else n
 tile=((e*(N//256)+p//256)*16+g//16)*4+p%256//64
 ix=g%16//4*256+p%64//16*64+p%8*8+g%4*2+p//8%2
 return raw[tile*9216+8192+ix]

def direct(wgs,N,up):
 ecount=wgs.nbytes//(N*4096*9//16);shape=(ecount,N//256,64,16,576);size=np.prod(shape)
 guard=np.full(int(size)+32,0xa5,np.uint8);out=guard[16:-16].reshape(shape);words=wgs.reshape(-1).view(np.uint32);raw=wgs.reshape(-1)
 units=ecount*(N//256)*64*16
 for first in range(0,units*128,32768):
  z=np.arange(first,min(first+32768,units*128));u=z//128;w=z%128;g=w//16;q=w//4%4;kk=w%4
  tile=u%16;C=u//16%64;G=u//1024%(N//256);e=u//(1024*(N//256))
  packed=np.zeros(len(z),np.uint32)
  for m in range(4):
   rr=m%2;half=m//2;n=canonical_row(G*256+tile*16+g+rr*8,up)
   for h in range(2):
    k=C*64+16*q+4*kk+2*half+h;c=source_code(words,e,n,k,N,up)
    for b,pos in enumerate(POS[m][h]):packed|=((c>>b)&1)<<pos
  # Output units have576B, but128code words occupy their first512B only.
  out.reshape(-1).view(np.uint32)[u*144+w]=packed
 for first in range(0,units*64,65536):
  z=np.arange(first,min(first+65536,units*64));u=z//64;v=z%64;g=v//8;q=v//2%4;rr=v%2
  tile=u%16;C=u//16%64;G=u//1024%(N//256);e=u//(1024*(N//256));n=canonical_row(G*256+tile*16+g+rr*8,up)
  out.reshape(-1)[u*576+512+v]=source_scale(raw,e,n,C*4+q,N,up)
 assert np.all(guard[:16]==0xa5) and np.all(guard[-16:]==0xa5)
 return out

def inverse(raw,N,up):
 ecount=raw.nbytes//(N*4096*9//16);source=raw.reshape(-1);words=source.view(np.uint32)
 rev=np.argsort(canonical_row(np.arange(N),up));outq=np.empty((ecount,N,4096),np.uint8);outs=np.empty((ecount,N,256),np.uint8);pos=np.asarray(POS,np.uint32)
 for first in range(0,outq.size,65536):
  z=np.arange(first,min(first+65536,outq.size));e=z//(N*4096);n=z//4096%N;k=z%4096;r=rev[n]
  u=((e*(N//256)+r//256)*64+k//64)*16+r%256//16
  w=r%8*16+k%64//16*4+k%16//4;m=r//8%2+2*(k//2%2);h=k%2
  packed=words[u*144+w];c=np.zeros(len(z),np.uint32)
  for b in range(4):c|=((packed>>pos[m,h,b])&1)<<b
  outq.reshape(-1)[z]=c
 for first in range(0,outs.size,65536):
  z=np.arange(first,min(first+65536,outs.size));e=z//(N*256);n=z//256%N;g=z%256;r=rev[n]
  u=((e*(N//256)+r//256)*64+g//4)*16+r%256//16
  outs.reshape(-1)[z]=source[u*576+512+r%8*8+g%4*2+r//8%2]
 return outq,outs

results=[]
for seed in (930433,930439):
 rng=np.random.default_rng(seed)
 for N in (4096,8192):
  q=rng.integers(0,16,(1,N,4096),dtype=np.uint8);sf=np.arange(N*256,dtype=np.uint32).astype(np.uint8).reshape(1,N,256);rng.shuffle(sf,axis=1)
  wgs=wgs_forward(q,sf,N==8192);before=hashlib.sha256(wgs.tobytes()).hexdigest()
  expected=forward(q,sf,N==8192);actual=direct(wgs,N,N==8192)
  assert np.array_equal(actual,expected)
  restored_q,restored_sf=inverse(actual,N,N==8192)
  assert np.array_equal(restored_q,q) and np.array_equal(restored_sf,sf)
  assert before==hashlib.sha256(wgs.tobytes()).hexdigest()
  results.append(dict(seed=seed,N=N,nibbles=q.size,scale_bytes=sf.size,output_bytes=actual.nbytes,exact=True,all_scale_codes=bool(len(np.unique(sf))==256),source_unchanged=True,canaries=True,inverse_original_bytes_exact=True))
  del q,sf,wgs,expected,actual,restored_q,restored_sf
r=dict(passed=True,cases=results,source_py_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),layout='50800 bit-permuted codes and same unit ordering, RAW originalE4M3 bytes; no zeroing or normalization.',all_code_values_and_all_scale_bytes=True,cpu_only=True,forward_w13_rows_independent_of_direct_inverse=True,device_decoder_source_present_but_uncompiled=True,loader_integration=False,limitations=['Byte-layout proof only, no numerical claim for NaN/negative scales.','No GPU converter, graph ownership, conversion timing, or full-FFN validation.','This layout must never be passed to original normalized-scale50800 kernel.'])
(D/'layout_proof.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
