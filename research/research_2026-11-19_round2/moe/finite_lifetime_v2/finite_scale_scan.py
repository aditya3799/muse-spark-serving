"""One-time actual E4M3 scale-byte scan; never run in a captured forward."""
import torch
import triton
import triton.language as tl

@triton.jit
def _nonfinite(src,flag,L:tl.constexpr,B:tl.constexpr,WGS:tl.constexpr=False):
    i=tl.program_id(0)*B+tl.arange(0,B)
    address=i//1024*9216+8192+i%1024 if WGS else i
    b=tl.load(src+address,i<L,other=0)
    hit=tl.sum(((b&127)==127).to(tl.int32),axis=0)>0
    if hit:tl.atomic_or(flag,1,sem='relaxed')

def verify_scale_bytes(stage13,stage2):
    for s in (stage13,stage2):
        if (not s.is_cuda or s.device!=stage13.device or not s.is_contiguous() or
                s.dtype not in (torch.uint8,torch.float8_e4m3fn) or not s.numel()):
            raise ValueError('expected contiguous actual E4M3 scale bytes on one CUDA device')
    with torch.cuda.device(stage13.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('finite-scale proof must precede graph capture')
        flag=torch.zeros((),device=stage13.device,dtype=torch.int32)
        for s in (stage13,stage2):
            b=s.view(torch.uint8)
            _nonfinite[(triton.cdiv(b.numel(),4096),)](b,flag,b.numel(),4096)
        return flag.item()==0


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
