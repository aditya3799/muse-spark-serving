"""Independent literal-pack addresses and full finite-code operand oracle."""
from pathlib import Path
import ast,hashlib,json,struct,random
P=Path(__file__).resolve().parent;A=P.parent/'aux_speculation';ns={};t=ast.parse((A/'test_packed_addresses.py').read_text());exec(compile(ast.Module(body=[x for x in t.body if isinstance(x,ast.FunctionDef)],type_ignores=[]),'pack_oracle','exec'),ns)
f32=lambda b:struct.unpack('<f',struct.pack('<I',b))[0]
u32=lambda f:struct.unpack('<I',struct.pack('<f',f))[0]
def quarter(c):
 m=c&7;bits=0x3e000000+(m<<22) if m>=2 else 0x3e000000 if m==1 else 0
 return f32(bits|((c&8)<<28))
def scale(c):
 e=(c>>3)&15;m=c&7;bits=((e+120)<<23)|(m<<20)
 if e==0:bits=u32(m/512.)
 if e==15 and m==7:bits=0x7fc00000
 return f32(bits|((c&128)<<24))
q=[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.];operand_checks=0
for c in range(16):
 for s in range(256):
  if s&127==127:continue
  e=(s&127)>>3;m=s&7;refscale=(m/512. if e==0 else (1+m/8.)*2.**(e-7))*(-1. if s&128 else 1.)
  assert u32(quarter(c)*scale(s))==u32(q[c]*refscale*.25);operand_checks+=1
address_checks=0;quads=0
for n,k in [(64,64),(64,128),(128,64),(128,128)]:
 for src in [False,True]:
  srcref=ns['reference'](n,k,4,src);inverse={logical:physical for physical,logical in srcref.items()}
  for dst in [False,True]:
   dstref=ns['reference'](n,k,16,dst)
   for base in range(0,n*k,4):
    group,within=divmod(base,n*16);nb,rem=divmod(within,1024);lane,inner=divmod(rem,32);i3=inner//8;a=inner//4%2
    word=group*(n*2)+nb*128+lane*4+i3;sp=group*n+nb*64+(lane//4)*8+i3*2+(0 if dst else a)
    for j in range(4):
     nib=2*a+j//2+4*(j%2) if src==dst else 2*(j//2)+a+4*(j%2)
     row,col=dstref[base+j];expected_sp=(col//16)*n+(row//64)*64+(row%8)*8+(row%64)//8
     assert word*8+nib==inverse[(row,col)];assert sp+(1 if dst and j>=2 else 0)==expected_sp
     assert base//4*8+j*2==2*(base+j);address_checks+=1
    quads+=1
from native_materialize import render_materializer
for src in [False,True]:
 for dst in [False,True]:
  for loop in [False,True]:assert 'materialize_packed' in render_materializer(8192,4096,src,dst,loop)
r={'status':'CPU_NATIVE_FOUR_VALUE_MATERIALIZER_ORACLE_PASS','finite_operand_cases':operand_checks,'address_value_checks':address_checks,'quads':quads,'scope':'Source bit construction versus independent FP4/FP8 arithmetic; all signed finite codes, including negativezero. Literal pinned pack-loop map independently checks one uint32word and one/twoFP8scale loads for each4value uint2store.','grid_variants':['tiled256threads,4BF16values/thread','fixedSMcount*4 grid,256threads with gridstride loop'],'memory_contract':'InputpackedB4+FP8group scales read-only; separate8-bytealignedBF16 output. No scratch, sharedmemory orCTAbarriers. Explicituint32N*K<2^32. Oneexpert at a time; same stream serial weightconsumer lifetime remains caller-owned.','numerics':'Decodeq/4 thenmul.rn.f32 with exact finiteE4M3scale, cvt.rnBF16; this equals q*scale/4 for finite inputs. NaN payloads remain outside gate. Global/epilogue factor4 not touched.','source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'native_materialize.cuh',P/'native_materialize.py',Path(__file__),A/'test_packed_addresses.py']},'next':'Independent Aux review, then bounded native fullvalue correctness and bestTriton256/8 control timing if rootlease granted. No wholehelper gain claim.'}
(P/'NATIVE_MATERIALIZER_CPU_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'operands':operand_checks,'addresses':address_checks,'quads':quads}))
