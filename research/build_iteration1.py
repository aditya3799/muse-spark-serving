from pathlib import Path
import subprocess,difflib
root=Path(r'D:\OpenKernelChallenge\serving')
repo=root/'sglang'
out=root/'iterations'; out.mkdir(exist_ok=True)
rel='python/sglang/srt/models/muse_spark.py'
p=repo/rel
base=p.read_text(encoding='utf-8')
old='''        self._split2_hi = None
        self._split2_lo = None
'''
new='''        self._split2_weight = None
'''
assert old in base
s=base.replace(old,new,1)
s=s.replace('''        self._split2_hi = hi.t().contiguous()
        self._split2_lo = lo.t().contiguous()
''','''        # Share the input read and GEMM launch while retaining separate FP32
        # accumulators for the high and low BF16 components.
        self._split2_weight = torch.cat((hi, lo)).t().contiguous()
''',1)
s=s.replace('''        acc = torch.mm(x, self._split2_hi, out_dtype=torch.float32)
        acc += torch.mm(x, self._split2_lo, out_dtype=torch.float32)
        return acc
''','''        acc = torch.mm(x, self._split2_weight, out_dtype=torch.float32)
        hi, lo = acc.chunk(2, dim=-1)
        return hi + lo
''',1)
assert s != base
p.write_text(s,encoding='utf-8',newline='\n')
patch=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
(out/'01_packed_router.diff').write_text(patch,encoding='utf-8')
print('Iteration 1: packed hi/lo router GEMM;',len(patch.encode()),'patch bytes')
