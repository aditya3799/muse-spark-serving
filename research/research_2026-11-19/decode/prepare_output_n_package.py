"""Package source only; do not activate or modify the frozen serving checkout."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent;old=here.parents[1]/'research_2026-11-18_lt_audit'
package=here/'output_n_package';package.mkdir(exist_ok=True)
adapter=(here/'output_n_adapter.py').read_text().replace('from runtime_projection import','from .runtime_projection import')
(package/'adapter.py').write_text(adapter)
runtime=(old/'runtime_projection.py').read_text()
runtime=runtime.replace('import torch\n','import torch\nfrom .native_build import native_library_path\n')
runtime=runtime.replace('str(HERE/"runtime_lt.so")','str(native_library_path())')
runtime=runtime.replace('    examples=list(examples)\n',
 '''    # Native PyTorch restricts Lt reduction schemes under stricter BF16
    # policies. This bridge intentionally supports only the tested native
    # default; do not silently re-enable reduced-precision partial reductions.
    if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is not True:
        return None,{"reason":"native BF16 reduction policy requires an unsupported Lt preference mask"}
    examples=list(examples)
''')
runtime=runtime.replace('("projection_lt_version",[],size),','("projection_lt_version",[],size),("projection_config",[ptr,num,num],num),')
runtime=runtime.replace('return dict(zip(("vendor_ordinal","local_workspace","full_workspace"),\n                    [self.lib.projection_attr(self.handle,index,i) for i in range(3)]))',
 '''return dict(zip(("vendor_ordinal","local_workspace","full_workspace"),
                    [self.lib.projection_attr(self.handle,index,i) for i in range(3)]),
                    attrs=dict(zip("id tile split reduction swizzle custom stages inner cluster".split(),
                    [self.lib.projection_config(self.handle,index,i) for i in range(9)])))''')
(package/'runtime_projection.py').write_text(runtime)
native=(old/'runtime_lt.cc').read_text()
native+='''
extern "C" int projection_config(void* q,int index,int field) {
 static const cublasLtMatmulAlgoConfigAttributes_t fields[]={
  CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,CUBLASLT_ALGO_CONFIG_STAGES_ID,
  CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
 if(field<0 || field>=9)return -1;
 auto* p=static_cast<Choice*>(q);int value=-1;
 return cublasLtMatmulAlgoConfigGetAttribute(&p->items.at(index).algo,fields[field],&value,sizeof(value),nullptr)==CUBLAS_STATUS_SUCCESS?value:-1;
}
'''
(package/'runtime_lt.cc').write_text(native)
(package/'__init__.py').write_text('from .entry import create_projection\n')
files={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*') if p.is_file() and p.suffix in ('.py','.cc')}
(package/'source_manifest.json').write_text(json.dumps(files,indent=2))
print(json.dumps(files,indent=2))
