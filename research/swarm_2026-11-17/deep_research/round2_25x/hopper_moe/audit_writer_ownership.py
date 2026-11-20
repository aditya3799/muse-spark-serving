from pathlib import Path
import json,collections
P=Path(__file__).resolve().parent;OLD=P.parents[1]/'hopper_moe'
rows=[]
configs=[('SM86_'+j['label'],j['configuration']) for j in json.loads((P/'SM86_REUSE_JOBS.json').read_text())['jobs'] if j['enabled']]
configs += [('SM90_'+j['job_id'],j['configuration']) for j in json.loads((OLD/'virtual_k_jobs.json').read_text())['jobs'] if j['policy']=='virtual']
for label,c in configs:
 bm,bn,bk=c['block_shape'];wm,wn,wk=c['warp_shape'];parts=bk//wk;physical=bm//wm*(bn//wn)*32;wgmma=c['mma_type']=='wgmma';expected=bm*bn//2
 for base in range(8):
  good=[];bad=[]
  for thread in range(physical):
   warp,lane=divmod(thread,32);mw=warp//(bn//wn)%(bm//wm);nw=warp%(bn//wn)
   if wgmma:outer_m=wn*4//64;outer_n=1;inner_m=2;inner_n=wm//8
   else:outer_m=wm//16;outer_n=wn//8;inner_m=2;inner_n=1
   for i in range(outer_m):
    for j in range(outer_n):
     for im in range(inner_m):
      for inn in range(inner_n):
       rowblock=i*inner_m+im;colblock=j*inner_n+inn
       if wgmma:
        subrow=(lane%4)*2+(lane%8)//4;row=mw*wm+8*colblock+subrow;count=64//wn
        col1=((nw%count*(8//count)+rowblock)^((subrow+base)%8))*4+lane//8
        idx=row*32+col1+(nw//count)*(bm*64//2)
       else:
        subrow=lane//4;row=mw*wm+8*rowblock+subrow;col=colblock*4+wn//2*nw
        row+=bm*(col//32);col=((col%32//4)^((subrow+base)%8))*4+lane%4;idx=row*32+col
       assert 0<=idx<expected,(label,idx,expected)
       good.append(idx)
       if thread<physical//parts:bad.append(idx)
  assert len(good)==expected and set(good)==set(range(expected)) and max(collections.Counter(good).values())==1
  assert len(set(bad))<expected,'Negative original ownership predicate did not expose missing output'
  # Canonical global-writer loop, including predicates and int4 swizzle, must consume only initialized half2 cells and write each valid output vector exactly once.
  slots=bm+7;route=[r+3 if r%3 else slots for r in range(bm)];n=bn*2;addresses=[]
  for tile_n in range(2):
   total=bm*bn//8
   for thread in range(physical):
    for it in range((total+physical-1)//physical):
     off=thread+physical*it
     if off>=total:continue
     sr,sc=divmod(off,8);src=(sr*8+(sc^((sr+base)%8)))*4
     assert all(src+d in set(good) for d in range(4))
     gr=route[sr%bm];gc=sr//bm*8+sc
     if gr>=slots:continue
     dst=tile_n*(bn//8)+gr*(n//8)+gc
     assert 0<=dst<slots*n//8;addresses.append(dst)
  expected_gmem={r*(n//8)+col for r in route if r<slots for col in range(n//8)}
  assert set(addresses)==expected_gmem and len(addresses)==len(expected_gmem)
 rows.append({'label':label,'physical_threads':physical,'half2_cells_per_tile':expected,'bases_checked':8,'buggy_predicate_half2_cells':len(set(bad)),'correct_writer_all_cells_exactly_once':True,'global_writer_predicates_swizzle_and_addresses_pass':True})
r={'status':'actual_epilogue_predicate_and_address_models_passed','rows':rows,'negative_control':'Original K_WARPS=BK/WK predicate suppresses required physical writers in every virtual configuration; caught before GPU execution.','source_fix':'virtual_k_smem_writer.cuh uses K_WARPS=Ctx::K_WARPS. Layout, conversion, shuffle, write split and global index equations are unchanged.','limits':'Models actual source formulas and predicates, not GPU instruction execution or numerical equivalence.'}
(P/'WRITER_OWNERSHIP_CORRECTION_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
