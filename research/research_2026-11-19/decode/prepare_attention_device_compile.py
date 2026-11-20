"""Extract device-only attention to qualify SM90 syntax without PyTorch headers.

The wrapper instantiates one kernel; it does not execute it. A successful
compile is not a correctness or performance qualification.
"""
import ast,hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
source=(here/'precision_attention.py').read_text()
tree=ast.parse(source)
cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
cuda=cuda[:cuda.index('template <int L>\nvoid launch(')]
cuda=cuda.replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <cmath>\n#include <algorithm>')
cuda=cuda.replace('#include <ATen/cuda/CUDAContext.h>','')
for name,flat,mode in [('request',False,1),('flat',True,0)]:
 wrapper=f'''\nextern "C" void compile_only_{name}(Params p, cudaStream_t stream) {{
  attn_kernel<8, {str(flat).lower()}, {mode}><<<dim3(1), Cfg<8, {str(flat).lower()}>::NTX, sizeof(Smem<8, {str(flat).lower()}>) + 1024, stream>>>(p);
}}\n'''
 path=here/f'precision_attention_{name}.cu';path.write_text(cuda+wrapper)
 print(path.name,path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest())
