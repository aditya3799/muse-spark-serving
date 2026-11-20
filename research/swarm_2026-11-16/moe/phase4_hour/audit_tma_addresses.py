"""CPU address-equivalence oracle for pinned canonical WGMMA TMA B/BS loaders."""
from pathlib import Path
import hashlib,json,itertools
P=Path(__file__).resolve().parent
manifest=json.loads((P/'producer_compile_jobs.json').read_text())
unique={}
for job in manifest['jobs']:
 c=job['configuration'];key=(c['shape_n'],c['shape_k'],*c['block_shape'][1:]);unique[key]=c
cases=[]
for (N,K,BN,BK),c in unique.items():
 for e,nb,kb in itertools.product((0,31),(0,N//BN-1),(0,K//BK-1)):
  for scale in (False,True):
   if scale:
    pack=16; cols=N//16;tilecols=BN//16;rows=BK//16;unitbytes=1
    legacystride=N//16;legacysmstride=BN//16
    base=e*(N//16)*(K//16)+kb*(BK//16)*(N//16)+nb*(BN//16)
   else:
    pack=16;cols=N//8;tilecols=BN//8;rows=BK//16;unitbytes=4
    legacystride=N//2;legacysmstride=BN//2
    base=e*N*K//32+kb*(BK//16)*(N//2)+nb*(BN//2)
   units_per_int4=16//unitbytes
   tma=[((e*(K//16)+kb*(BK//16)+z)*cols+nb*tilecols+y)*pack+x for z in range(rows) for y in range(tilecols) for x in range(pack)]
   legacy=[(base+(q//legacysmstride)*legacystride+q%legacysmstride)*units_per_int4+lane for q in range(rows*legacysmstride) for lane in range(units_per_int4)]
   assert len(tma)==len(legacy)==BN*BK*(1 if scale else 2)//16
   assert tma==legacy
   assert max(tma)<32*N*K*(1 if scale else 2)//16
   cases.append({'N':N,'K':K,'BN':BN,'BK':BK,'expert':e,'n_block':nb,'k_block':kb,'tensor':'scale' if scale else 'code','equal_units':len(tma),'unit_bytes':unitbytes})
v=P.parent/'humming_vendor/humming';files=['csrc/launcher/tensor.h','include/humming/memory/g2s_loader/loader_b.cuh','include/humming/memory/g2s_loader/loader_bs.cuh']
r={'status':'PASS','checks':len(cases),'unique_transfer_geometries':len(unique),'cases':cases,'source_hashes':{f:hashlib.sha256((v/f).read_bytes()).hexdigest() for f in files},'scope':'CPU address model: canonical contiguous WGMMA NVFP4/BF16/FP8 group16, no packedK, no multicast. Entire destination tiles atfirst/last expert/N/K coordinates equal between pinned legacy int4 copy and 3D TMA descriptor coordinates. Does not execute CUDA descriptor encoding, barriers, SM90 instructions or production GEMM.'}
(P/'tma_address_audit.json').write_bytes((json.dumps(r,indent=2)+'\n').encode())
print('PASS',len(cases),'address tiles across',len(unique),'transfer geometries')
