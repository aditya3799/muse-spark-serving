from pathlib import Path
import ast
here=Path(__file__).resolve().parent;p=here/'finite_scale_scan.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace('L:tl.constexpr,B:tl.constexpr):','L:tl.constexpr,B:tl.constexpr,WGS:tl.constexpr=False):')
s=s.replace('b=tl.load(src+i,i<L,other=0)','address=i//1024*9216+8192+i%1024 if WGS else i\n    b=tl.load(src+address,i<L,other=0)')
s+='''

def verify_resident_scale_bytes(resident):
    """Scan the actual packed source SF regions, without retaining raw weights."""
    a,b=resident.w13,resident.w2
    e=a.numel()//(8192*2304)
    if (not 1<=e<=32 or a.numel()!=e*8192*2304 or b.numel()!=e*4096*2304 or
            any(x.dtype!=torch.uint8 or not x.is_cuda or not x.is_contiguous()
                or x.device!=a.device for x in (a,b))):
        raise ValueError('invalid resident WGS stage storage')
    with torch.cuda.device(a.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('finite-scale proof must precede graph capture')
        flag=torch.zeros((),device=a.device,dtype=torch.int32)
        for stage in (a,b):
            scales=stage.numel()//9
            _nonfinite[(triton.cdiv(scales,4096),)](stage,flag,scales,4096,True)
        return flag.item()==0
'''
ast.parse(s);p.write_text(s,encoding='utf-8')
p=here/'test_finite_scale_scan.py';s=p.read_text(encoding='utf-8-sig').replace('from finite_scale_scan import verify_scale_bytes','from finite_scale_scan import verify_scale_bytes,verify_resident_scale_bytes')
needle="report=dict(complete=True"
add='''# Actual WGS scale-region addressing: arbitrary code bytes including127/255
# must not trigger the finite-SF gate; each packed scale region is scanned.
class Resident:pass
resident=Resident()
for name,n in (('w13',8192),('w2',4096)):
 stage=torch.randint(0,256,(n*2304,),device='cuda',dtype=torch.uint8)
 stage.view(-1,9216)[:,8192:].fill_(56)
 setattr(resident,name,stage)
assert verify_resident_scale_bytes(resident)
for name in ('w13','w2'):
 stage=getattr(resident,name)
 for at in (8192,9215,stage.numel()-1024,stage.numel()-1):
  for bad in (127,255):
   stage[at]=bad;assert not verify_resident_scale_bytes(resident)
   rows.append(dict(packed_stage=name,offset=at,bad=bad,rejected=True))
  stage[at]=56
 assert verify_resident_scale_bytes(resident)
'''
s=s.replace(needle,add+needle);ast.parse(s);p.write_text(s,encoding='utf-8')
print('Added actual packed-SF scan and negative addressing probes')
