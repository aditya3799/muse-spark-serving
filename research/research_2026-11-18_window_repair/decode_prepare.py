from pathlib import Path
p=Path(__file__).with_name("decode_bench.py");s=p.read_text(encoding="utf-8-sig").replace('def prepare(seed):','def prepare(seed,mode):')
s=s.replace('weights=(rand((v,r),.2),rand((v,r),.2),rand((r,h),h**-.5))','weights=(rand((v,r),.2),rand((v,r),.2),rand((r,h),h**-.5)) if mode!="walk" else (None,None,None)')
s=s.replace('g,weights,model=prepare(seed)','g,weights,model=prepare(seed,mode)')
p.write_text(s,encoding="utf8")
p=Path(__file__).with_name("decode_build_packed.py");s=p.read_text(encoding="utf-8-sig")
s=s.replace('(H/"decode_packed_kernels.py").write_text(kernel,encoding="utf8")','(D/"packed_kernels_checked.py").write_text(kernel,encoding="utf8")')
p.write_text(s,encoding="utf8")
p=Path(__file__).with_name("decode_packed_parity.py");s=p.read_text(encoding="utf-8-sig").replace('v[:,1:4]=v[:,:1]','v[:,1:4]=v[:,:1].clone()');p.write_text(s,encoding="utf8")

