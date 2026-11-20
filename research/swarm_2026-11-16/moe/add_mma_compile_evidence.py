from pathlib import Path
import ast
p=Path(__file__).resolve().parent/'offline_sm90_compile.py';s=p.read_text(encoding='utf-8')
s=s.replace('import argparse,hashlib,json,os,sys,time','import argparse,hashlib,json,os,re,sys,time')
anchor="assert not kernel.cubin_loaded and not torch.cuda.is_initialized()\n(a.out/'kernel.cu')"
replacement="""assert not kernel.cubin_loaded and not torch.cuda.is_initialized()
mma=kernel.mma_op_class
assert mma.cd_dtype=='f32' and mma.val_type_cd=='float'
mma_cpp=mma.to_cpp_str()
(a.out/'mma_op_class.cuh').write_text(mma_cpp,encoding='utf-8')
(a.out/'kernel.cu')"""
assert anchor in s;s=s.replace(anchor,replacement)
s=s.replace("'mma_type':str(kernel.mma_type),", "'mma_type':str(kernel.mma_type),'actual_mma_op_class':{'python_class':type(mma).__name__,'shape':mma.shape,'a_dtype':mma.a_dtype,'b_dtype':mma.b_dtype,'accumulator_dtype':mma.cd_dtype,'accumulator_cpp_type':mma.val_type_cd,'cpp_sha256':hashlib.sha256(mma_cpp.encode()).hexdigest(),'ptx_operations':sorted(set(re.findall(r'(?:wgmma\\.mma_async|mma\\.sync)[^ \\\"\\n;]+',mma_cpp)))},")
ast.parse(s);p.write_bytes(s.encode('utf-8'));print('actual MMA class and PTX reporting added')
