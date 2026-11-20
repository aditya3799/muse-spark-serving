from pathlib import Path
import ast,json,sys
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));from virtual_k_codegen import render_virtual_k
jobs=json.loads((P/'SM86_REUSE_JOBS.json').read_text())['jobs'];checks={'A_logical_fragment_vectors':0,'B_byte_offsets':0,'scale_byte_offsets':0,'K_partition_sequences':0,'output_row_ownership':0}
baseline=jobs[0]['configuration'];bm0,bn,bk=baseline['block_shape'];wm,wn,wk=baseline['warp_shape'];parts=bk//wk;iters=wk//16
assert (bm0,wm,wn,wk,parts)==(16,16,64,64,2)
for job in jobs[1:]:
 c=job['configuration'];bm=c['block_shape'][0];mwarps=bm//wm;nwarps=bn//wn
 owned=[]
 for mw in range(mwarps):
  for nw in range(nwarps):
   warp=mw*nwarps+nw
   assert warp%nwarps==nw and warp//nwarps%mwarps==mw
   for part in range(parts):
    old_seq=[base+part*wk+i*16 for base in range(0,c['shape_k'],bk) for i in range(iters)]
    new_seq=[base+v*16 for base in range(0,c['shape_k'],bk) for v in range(iters*parts) if v//iters==part]
    assert old_seq==new_seq;checks['K_partition_sequences']+=1
    for i in range(iters):
     v=part*iters+i
     for lane in range(32):
      old_b=(i*bn*16//2)+(nwarps*32*iters*part+32*nw+lane)*16
      new_b=v*bn*16//2+(32*nw+lane)*16
      assert old_b==new_b;checks['B_byte_offsets']+=1
      old_bs=(part*wk//16+i)*bn+(lane//4+nw*8)*8
      new_bs=v*bn+(lane//4+nw*8)*8
      assert old_bs==new_bs;checks['scale_byte_offsets']+=1
      for smem_base in range(8):
       logical_row=mw*wm+lane%16;logical_kvec=part*iters*2+i*2+lane//16
       row=bm*(logical_kvec//8)+logical_row;col=(logical_kvec%8)^((row+smem_base)%8)
       recovered_row=row%bm;recovered_kvec=(row//bm)*8+(col^((row+smem_base)%8))
       assert (recovered_row,recovered_kvec)==(logical_row,logical_kvec)
       old_row=lane%16;old_kvec=part*iters*2+i*2+lane//16
       assert (old_row+mw*wm,old_kvec)==(logical_row,logical_kvec);checks['A_logical_fragment_vectors']+=1
   for r in range(wm):
    for n in range(wn):owned.append((mw*wm+r,nw*wn+n))
 assert len(owned)==len(set(owned))==bm*bn;checks['output_row_ownership']+=len(owned)
 assert c['warp_shape']==baseline['warp_shape'] and c['block_shape'][1:]==baseline['block_shape'][1:]
for file in ('build_reuse_prototype.py','virtual_k_codegen.py'):ast.parse((P/file).read_text(encoding='utf-8-sig'))
wmma=(P/'virtual_k_wmma.cuh').read_text();assert 'regs_c[virtual_part][m][j], regs_c[virtual_part][m][j]' in wmma
assert wmma.count('void reduce_virtual_partitions()')==1 and 'ElementA::kBits == 16' in wmma
report={'status':'CPU_mapping_passed','checks':checks,'K_tree':'Two original independent FP32 banks; reduce p0+p1 once after full K, then unchanged BF16 epilogue.','M2_reuse':'One staged B/scales tile serves two M warps; primitive warpM/N/K unchanged. Uniform sparse experts may gain no bytes and incur extra padded MMA; hot or crowded experts can halve requested B/scale loads.','limits':'Source/integer-model evidence only; native SM86 output equivalence, resource pressure and performance still require execution. No H200 evidence.'}
(P/'CPU_REUSE_CONTRACT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
