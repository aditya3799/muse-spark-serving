"""Literal sourcepack/destinationpack and canonical-scatter proof for outputN bands."""
import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;source=P/'test_packed_addresses.py';tree=ast.parse(source.read_text());ns={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(source),'exec'),ns)
checks=0;scalechecks=0;negative=0;cases=[]
for full_n,k in ((256,64),(512,128)):
 for tile_n in (128,256):
  for src in (False,True):
   sr=ns['reference'](full_n,k,4,src);inv={v:key for key,v in sr.items()}
   for dst in (False,True):
    dr=ns['reference'](tile_n,k,16,dst)
    for off in sorted(set((0,64,full_n-tile_n))):
     if off+tile_n>full_n:continue
     wrong=0
     for base in range(0,tile_n*k,4):
      group,within=divmod(base,tile_n*16);nb,rem=divmod(within,1024);lane,inner=divmod(rem,32);i3=inner//8;a=inner//4%2;source_nb=nb+off//64
      word=group*full_n*2+source_nb*128+lane*4+i3;sp=group*full_n+source_nb*64+(lane//4)*8+i3*2+(0 if dst else a)
      for j in range(4):
       nib=2*a+j//2+4*(j%2) if src==dst else 2*(j//2)+a+4*(j%2);row,col=dr[base+j];row+=off
       assert word*8+nib==inv[(row,col)];checks+=1
       expected=(col//16)*full_n+(row//64)*64+(row%8)*8+(row%64)//8
       assert sp+(1 if dst and j>=2 else 0)==expected;scalechecks+=1
       wrong+=(group*tile_n*2+source_nb*128+lane*4+i3)*8+nib!=inv[(row,col)]
     if full_n!=tile_n:assert wrong>0;negative+=1
     cases.append({'sourceN':full_n,'targetN':tile_n,'K':k,'offset':off,'sourceTranspose':src,'targetTranspose':dst,'naive_target_stride_mismatches':wrong})
scatterchecks=0
for full_n in (4096,8192):
 for tile_n in (128,256):
  for rows in (0,1,7):
   seen=set()
   for off in range(0,full_n,tile_n):
    for row in range(rows):
     for col in range(tile_n):
      index=row*full_n+off+col;assert index not in seen and 0<=index<rows*full_n;seen.add(index);scatterchecks+=1
   assert len(seen)==rows*full_n
from native_materialize_tiled import render_materializer
for full_n in (4096,8192):
 for tile_n in (128,256):
  for src in (False,True):
   for dst in (False,True):
    code=render_materializer(tile_n,4096,src,dst,False,full_n);assert '@' not in code and f'#define SOURCE_N {full_n}u' in code
r={'status':'CPU_OUTPUT_N_BAND_LAYOUT_PASS','weight_address_checks':checks,'scale_address_checks':scalechecks,'wrong_stride_negative_cases':negative,'scatter_coverage_checks':scatterchecks,'cases':cases,'scope':'Different sourceN andtargetN,stride-safe64-aligned rowoffset, fullK retained, allsrc/dstlayout combinations, exact canonical outputcolumn partition with nooverlap/omission. No native compilation/runtime.','source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),source,P/'native_materialize_tiled.py',P/'native_materialize_tiled.cuh')}}
(P/'OUTPUT_N_BAND_CPU_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('status','weight_address_checks','scale_address_checks','wrong_stride_negative_cases','scatter_coverage_checks')}))
