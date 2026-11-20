from pathlib import Path
import ast
here=Path(__file__).resolve().parent;out=here/'finite_lifetime_v2'
s=(here/'finite_scale_scan.py').read_text(encoding='utf-8-sig')
a=s.index('def verify_scale_bytes(');b=s.index('\ndef verify_resident_scale_bytes',a)
new='''def verify_scale_bytes(stage13,stage2):
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

'''
s=s[:a]+new+s[b:];ast.parse(s);(out/'finite_scale_scan.py').write_text(s,encoding='utf-8')
print('Device-context guard fixed in v2 only; resident scanner numerical source unchanged')
