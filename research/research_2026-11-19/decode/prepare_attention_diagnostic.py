"""Generate a small C ABI for a Hopper differential diagnostic; no execution."""
import ast,hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
tree=ast.parse((here/'precision_attention.py').read_text())
cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
cuda=cuda[:cuda.index('template <int L>\nvoid launch(')]
cuda=cuda.replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <cmath>\n#include <algorithm>\n#include <cstddef>')
cuda=cuda.replace('#include <ATen/cuda/CUDAContext.h>','')
fields='qkv seq_lens cache_loc k_cache v_cache page_table k_scale v_scale inv_freq positions out ws_acc ws_l ws_cnt B Hq Hkv P max_pages window rope n_splits flat nc dbg cs ncl ld_qkv row_bytes scale_log2 coop lp lrb off32 ns_magic'.split()
cuda+='\nextern "C" size_t attention_params_size() { return sizeof(Params); }\n'
cuda+='extern "C" size_t attention_param_offset(int i) {\n static const size_t offsets[]={'+','.join('offsetof(Params,'+f+')' for f in fields)+'};\n return i>=0 && i<'+str(len(fields))+' ? offsets[i] : size_t(-1);\n}\n'
cuda+='''
extern "C" int attention_diagnostic(const void* raw, cudaStream_t stream) {
 const Params* p=static_cast<const Params*>(raw);
 if (!p || p->B<=0 || p->Hq!=16 || p->Hkv!=2 || p->P!=1 || p->flat ||
     p->coop || p->cs || p->nc!=1 || p->n_splits<1 || p->n_splits>1024)
   return int(cudaErrorInvalidValue);
 constexpr int smem=sizeof(Smem<8,false>)+1024;
 auto kernel=attn_kernel<8,false,1>;
 cudaError_t error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 if(error!=cudaSuccess)return int(error);
 kernel<<<dim3(p->n_splits,p->B,1),Cfg<8,false>::NTX,smem,stream>>>(*p);
 return int(cudaGetLastError());
}
'''
path=here/'attention_diagnostic_native.cu';path.write_text(cuda)
(here/'attention_diagnostic_source.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 variant_sha256=hashlib.sha256((here/'precision_attention.py').read_bytes()).hexdigest(),
 scope='Device-only q8 plain per-request diagnostic. Not integrated or executed; output initialized to zero by Python for padded requests.',fields=fields),indent=2))
print(path.name,path.stat().st_size)
