"""Read CUDA ELF constants/sections with standard Python; no module loading."""
import struct
from pathlib import Path
def cuda_elf_constants(path):
 data=Path(path).read_bytes();head=struct.unpack_from('<16sHHIQQQIHHHHHH',data,0)
 assert head[0][:6]==b'\x7fELF\x02\x01'
 shoff,shsize,shcount,shstr=head[6],head[11],head[12],head[13]
 sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*shsize) for i in range(shcount)]
 strings=sections[shstr];names=data[strings[4]:strings[4]+strings[5]]
 def name_at(blob,start):return blob[start:blob.index(b'\0',start)].decode()
 named=[name_at(names,s[0]) for s in sections];wanted={'NUM_THREADS','NUM_MATH_THREADS','NUM_LOAD_THREADS','SMEM_SIZE','SMEM_SIZE_A','SMEM_SIZE_B','SMEM_SIZE_REDUCE','BLOCK_SHAPE_K','WARP_SHAPE_K','NUM_CTAS_PER_SM'};values={}
 for s in sections:
  if s[1]!=2:continue
  st=sections[s[6]];strings=data[st[4]:st[4]+st[5]]
  for off in range(s[4],s[4]+s[5],s[9]):
   nm,info,other,index,value,size=struct.unpack_from('<IBBHQQ',data,off)
   name=name_at(strings,nm)
   if name not in wanted:continue
   assert size==4 and index<len(sections),(name,size,index)
   sec=sections[index];values[name]=struct.unpack_from('<I',data,sec[4]+value-sec[3])[0]
 shared={n:s[5] for n,s in zip(named,sections) if n.startswith('.nv.shared.') and not n.startswith('.nv.shared.reserved')}
 return {'constants':values,'static_shared_sections':shared,'static_shared_bytes':sum(shared.values()),'static_shared_reserved_sections':{n:s[5] for n,s in zip(named,sections) if n.startswith('.nv.shared.reserved')}}
