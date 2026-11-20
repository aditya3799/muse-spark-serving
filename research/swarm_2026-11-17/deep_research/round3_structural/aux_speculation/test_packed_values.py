"""Value-level CPU oracle for direct packed staging, independent literal pack loops."""
import ast,json,math,struct,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;s=p/'test_packed_addresses.py';tree=ast.parse(s.read_text());ns={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(s),'exec'),ns)
qvals=[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]
def scale_value(byte):
 e=(byte>>3)&15;m=byte&7
 return m/512 if e==0 else (1+m/8)*2.**(e-7)
def code(n,k):return (n*3+k*5+k//16*7+n//8)%16
def scale(n,g):return (n*17+g*29+n//8)%127
bits=lambda x:struct.pack('<d',x)
checks=0;negatives=0;cases=[]
for n,k in ((64,128),(128,64)):
 for src_transpose in (False,True):
  sr=ns['reference'](n,k,4,src_transpose);packed=[0]*(n*k//2)
  for pos,(row,col) in sr.items():packed[pos//2]|=code(row,col)<<((pos%2)*4)
  scalepacked=[None]*(n*k//16)
  for row in range(n):
   for group in range(k//16):scalepacked[group*n+(row//64)*64+(row%8)*8+(row%64)//8]=scale(row,group)
  assert None not in scalepacked
  for dst_transpose in (False,True):
   expected=ns['reference'](n,k,16,dst_transpose);wrong=0
   for pos,(rn,ck) in expected.items():
    row,col=ns['inverse_bf16'](n,pos,dst_transpose);assert(row,col)==(rn,ck)
    nib=ns['direct'](n,k,row,col,4,src_transpose);q=(packed[nib//2]>>((nib%2)*4))&15
    scaleidx=(col//16)*n+(row//64)*64+(row%8)*8+(row%64)//8
    actual=qvals[q]*scale_value(scalepacked[scaleidx])*.25
    reference=qvals[code(rn,ck)]*scale_value(scale(rn,ck//16))*.25
    assert bits(actual)==bits(reference);checks+=1
    bad=ns['direct'](n,k,row,col,4,not src_transpose);badq=(packed[bad//2]>>((bad%2)*4))&15
    wrong+=bits(qvals[badq]*scale_value(scalepacked[scaleidx])*.25)!=bits(reference)
   assert wrong>0;negatives+=1;cases.append({'N':n,'K':k,'source_transpose':src_transpose,'target_transpose':dst_transpose,'incorrect_source_layout_mismatches':wrong})
r={'status':'CPU_PACKED_VALUE_ORACLE_PASS_NOT_GPU_VALIDATED','bitwise_value_checks':checks,'negative_controls_detected':negatives,'cases':cases,'source_hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (s,p/'packed_staged_ops.py',Path(__file__))},'scope':'Literal source and destination pack loops versus direct address inverse/gather, full finite positive scale encodings and signedzero; source/target layout modes independently varied. CPU binary arithmetic exact, not native kernel proof.'}
(p/'PACKED_VALUE_CPU_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'checks':checks,'negative_controls':negatives}))
