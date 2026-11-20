import pathlib
p=pathlib.Path('serving/research_2026-11-19_round2/moe/prove_pr50800_rawscale_layout.py');s=p.read_text(encoding='utf-8-sig')
insert='''
def inverse(raw,N,up):
 ecount=raw.nbytes//(N*4096*9//16);source=raw.reshape(-1);words=source.view(np.uint32)
 rev=np.argsort(canonical_row(np.arange(N),up));outq=np.empty((ecount,N,4096),np.uint8);outs=np.empty((ecount,N,256),np.uint8);pos=np.asarray(POS,np.uint32)
 for first in range(0,outq.size,65536):
  z=np.arange(first,min(first+65536,outq.size));e=z//(N*4096);n=z//4096%N;k=z%4096;r=rev[n]
  u=((e*(N//256)+r//256)*64+k//64)*16+r%256//16
  w=r%8*16+k%64//16*4+k%16//4;m=r//8%2+2*(k//2%2);h=k%2
  packed=words[u*144+w];c=np.zeros(len(z),np.uint32)
  for b in range(4):c|=((packed>>pos[m,h,b])&1)<<b
  outq.reshape(-1)[z]=c
 for first in range(0,outs.size,65536):
  z=np.arange(first,min(first+65536,outs.size));e=z//(N*256);n=z//256%N;g=z%256;r=rev[n]
  u=((e*(N//256)+r//256)*64+g//4)*16+r%256//16
  outs.reshape(-1)[z]=source[u*576+512+r%8*8+g%4*2+r//8%2]
 return outq,outs
'''
assert '\nresults=[]' in s;s=s.replace('\nresults=[]',insert+'\nresults=[]')
s=s.replace('  assert np.array_equal(actual,expected)','  assert np.array_equal(actual,expected)\n  restored_q,restored_sf=inverse(actual,N,N==8192)\n  assert np.array_equal(restored_q,q) and np.array_equal(restored_sf,sf)')
s=s.replace('source_unchanged=True,canaries=True','source_unchanged=True,canaries=True,inverse_original_bytes_exact=True')
s=s.replace('del q,sf,wgs,expected,actual','del q,sf,wgs,expected,actual,restored_q,restored_sf')
s=s.replace('device_decoder_for_raw_scales_not_implemented=True','device_decoder_source_present_but_uncompiled=True')
p.write_text(s)
