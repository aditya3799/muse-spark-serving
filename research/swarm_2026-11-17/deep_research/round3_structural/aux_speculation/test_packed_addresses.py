"""CPU port of pinned pack loops versus closed-form direct staging addresses."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
source=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming/kernel/process.cuh'
def reference(n,k,bits,transpose):
 out={}
 for bn in range(n//64):
  for bk in range(k//64):
   for lane in range(32):
    tmp=[None]*128
    for i in range(8):
     for j in range(8):
      for kk in range(2):
       row=i*8+lane//4;col=j*8+(lane%4)*2+kk
       i1=j//2;i3=i//2;i4=i%2;i5=j%2
       a,b=(i5,i4) if transpose else (i4,i5)
       idx=((((i1*4+i3)*2+a)*2+b)*2+kk)
       assert tmp[idx] is None;tmp[idx]=(bn*64+row,bk*64+col)
    # Literal structure of humming_pack_weight<BitsB,16> and final stores.
    packed=[];N=32//bits;stride=2;group=N//stride
    for i in range(4):
     for j in range(bits):
      word=[]
      for kk in range(N):
       new_k=(kk%group)*stride+kk//group
       word.append(tmp[i*32+j*N+new_k])
      packed.append(word)
    outstride=16*n*bits//32;coloffset=16*(64*bn)*bits//32
    for i in range(4):
     row=bk*4+i
     for kk in range(bits):
      word_index=row*outstride+coloffset+lane*bits+kk
      for sub,logical in enumerate(packed[i*bits+kk]):
       physical=word_index*(32//bits)+sub
       assert physical not in out;out[physical]=logical
 return out
def direct(n,k,row,col,bits,transpose):
 lane=(row%8)*4+(col%8)//2;i3=(row%64)//16;i4=(row//8)%2;i5=(col//8)%2
 a,b=(i5,i4) if transpose else (i4,i5);inner=((i3*2+a)*2+b)*2+col%2
 if bits==16:wordpart=inner//2;sub=inner%2
 else:
  wordpart=inner//8;s=inner%8;sub=s//2+4*(s%2)
 word=(col//16)*(n*16*bits//32)+(row//64)*(64*16*bits//32)+lane*bits+wordpart
 return word*(32//bits)+sub
def inverse_bf16(n,physical,transpose):
 ktile,within=divmod(physical,n*16);nb,rem=divmod(within,1024);lane,inner=divmod(rem,32)
 i3=inner//8;a=(inner//4)%2;b=(inner//2)%2;i4,i5=(b,a) if transpose else (a,b)
 return nb*64+i3*16+i4*8+lane//4,ktile*16+i5*8+(lane%4)*2+inner%2
checks=0;cases=[]
for n,k in ((64,64),(64,128),(128,64),(128,128)):
 for transpose in (False,True):
  for bits in (4,16):
   ref=reference(n,k,bits,transpose);assert len(ref)==n*k
   for physical,logical in ref.items():
    assert direct(n,k,*logical,bits,transpose)==physical
    if bits==16:assert inverse_bf16(n,physical,transpose)==logical
    checks+=1
   cases.append({'n':n,'k':k,'bits':bits,'wgmma_transpose':transpose,'values':n*k})
# Actual transformed group scale order: transpose[G,N], then 64-element permutation.
scale_checks=0
for n in (64,128):
 for groups in (4,8):
  perm=[x+i for i in range(8) for x in (0,8,16,24,32,40,48,56)]
  for group in range(groups):
   for row in range(n):
    phys=group*n+(row//64)*64+(row%8)*8+(row%64)//8
    assert group*n+(row//64)*64+perm[phys%n%64]==group*n+row
    scale_checks+=1
r={'status':'CPU_PACK_ADDRESS_PROOF_PASS','value_address_checks':checks,'group_scale_address_checks':scale_checks,'cases':cases,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source':str(source),'scope':'Independent literal pack-loop model versus closed-form forward/inverse address formulas for B4 andB16 with A16,interleave3,no zero point,unpadded aligned shapes. Includes SM86 no-miniblock-transpose and SM90 WGMMA transpose. No GPU execution or kernel arithmetic proof.'}
(P/'PACKED_STAGING_CPU_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'value_checks':checks,'scale_checks':scale_checks}))
