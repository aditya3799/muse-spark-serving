import json
from pathlib import Path
from collections import Counter
from canonical_tile_adapter import humming_code_address, humming_scale_address, wgs_word_coordinates, physical_row
rows=[]
for gu in (False,True):
 N=8192 if gu else 4096
 code_sectors=set();scale_sectors=set();word_counts=Counter();qstats=[]
 for q in range(4):
  qs=set(); ss=set()
  for grp in range(4):
   for warp in range(4):
    for lane in range(32):
     for j in range(4):
      addr=[humming_code_address(n,k,N,True)[0] for n,k in wgs_word_coordinates(0,0,q,grp,warp,lane,j,gu)]
      word_counts[len(set(addr))]+=1
      qs.update(a*4//128 for a in addr)
    for g in range(8):
     for j in range(4):
      for h in range(2):
       n=physical_row(q*64+warp*16+g+h*8,gu)
       ss.add(humming_scale_address(n,grp*4+j,N)//128)
  code_sectors.update(qs);scale_sectors.update(ss)
  qstats.append({'q':q,'code_sector_bytes':len(qs)*128,'scale_sector_bytes':len(ss)*128})
 rows.append({'gate_up':gu,'full_stage_code_sector_bytes':len(code_sectors)*128,'full_stage_scale_sector_bytes':len(scale_sectors)*128,'unique_source_words_per_donor_word':dict(word_counts),'per_q':qstats})
result={'scope':'Static 128-byte sector footprint only; no timing/cache/transaction claim. Diagnostic direct gather is too instruction-heavy to promote. Fast design must coalesce source copies then transpose in shared/registers.','stages':rows}
Path(__file__).with_name('canonical_tile_traffic.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
