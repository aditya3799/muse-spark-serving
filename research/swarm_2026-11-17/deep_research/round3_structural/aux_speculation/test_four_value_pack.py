"""Independent literal-pack oracle for four-value native materializer loads."""
import ast,json,struct,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;source=P/'test_packed_addresses.py';tree=ast.parse(source.read_text());ns={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(source),'exec'),ns)
q=[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]
code=lambda n,k:(n*3+k*5+k//16*7+n//8)%16
scale=lambda n,g:(n*17+g*29+n//8)%127
def sf(byte):
 e=byte//8;m=byte%8
 return m/512 if e==0 else (1+m/8)*2.**(e-7)
bits=lambda x:struct.pack('<d',x)
checks=0;quads=0;cases=[]
for N,K in ((64,128),(128,64)):
 for src in (False,True):
  packed=[0]*(N*K//8)
  for pos,(n,k) in ns['reference'](N,K,4,src).items():packed[pos//8]|=code(n,k)<<(4*(pos%8))
  for dst in (False,True):
   reference=ns['reference'](N,K,16,dst)
   for base in range(0,N*K,4):
    group,within=divmod(base,N*16);nb,rem=divmod(within,1024);lane,inner=divmod(rem,32);a=(inner//4)%2
    wordindex=group*2*N+nb*128+lane*4+inner//8;word=packed[wordindex];scales=set()
    for j in range(4):
     nibble=2*a+j//2+4*(j%2) if src==dst else 2*(j//2)+a+4*(j%2)
     qcode=(word>>(nibble*4))&15;n,k=reference[base+j];actual=q[qcode]*sf(scale(n,k//16))*.25;expected=q[code(n,k)]*sf(scale(n,k//16))*.25
     assert bits(actual)==bits(expected)
     # Literal pack-loop-derived logical coordinate independently locates scale.
     physical_scale=(k//16)*N+(n//64)*64+(n%8)*8+(n%64)//8;scales.add(physical_scale);checks+=1
    assert len(scales)==(2 if dst else 1);quads+=1
   cases.append({'N':N,'K':K,'source_transpose':src,'target_transpose':dst,'scale_loads_per_quad':2 if dst else 1})
r={'status':'CPU_FOUR_VALUE_PACK_ORACLE_PASS','value_checks':checks,'quads':quads,'cases':cases,'scope':'Oneuint32sourcewordperfouradjacentBF16outputs, exact nibbleorder forboth transposeflags independently, one/two scaleaddress requirement. NativeCUDA compiler/runtime stillneedsvalidation.','source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),source)}}
(P/'FOUR_VALUE_PACK_CPU_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'checks':checks,'quads':quads}))
