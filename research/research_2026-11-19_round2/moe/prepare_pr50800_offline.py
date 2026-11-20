import pathlib,hashlib,json
p=pathlib.Path(__file__).resolve().parent;s=p/'sources/nov10_sweep/pr50800/nvfp4_w4a16_sm90.cuh';raw=s.read_bytes();text=raw.decode();out=p/'pr50800_offline';out.mkdir(exist_ok=True)
assert hashlib.sha256(raw).hexdigest()=='92219b6456049529db50af189887b7a17132b3ce53a4375bf87912ea0f19f572'
start=text.index('namespace sglang::nvfp4_w4a16_sm90 {');end=text.index('// Token rows: BF16 [rows][cols]')
body=text[start:end]
header=text[:text.index('#pragma once')]
source=header+'#include <cuda_runtime.h>\n#include <cuda.h>\n#include <cuda_bf16.h>\n#include <algorithm>\n#include <cstdint>\n#include <type_traits>\n'+body+'\n} // namespace sglang::nvfp4_w4a16_sm90\n'
(out/'device_only.cu').write_text(source)
manifest=dict(upstream_head='81502f72823876ef01c1f273d7061e1705361735',upstream_sha256=hashlib.sha256(raw).hexdigest(),device_body_sha256=hashlib.sha256(body.encode()).hexdigest(),extracted_source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformation='Keep copyright, exact namespace/device body through kSmallRoutes; replace removed SGLang/TVM host includes with CUDA/C++ standard includes; omit host tensor-map/runner wrappers only.',expected_global_kernels=['grouped_k','pair_k','count_k','plan_k','scatter_k','front_small_k','gather_k','combine_k'],device_math_unchanged=True)
(out/'extraction.json').write_text(json.dumps(manifest,indent=2))
original=(p/'compile_wgs_split_dependency.py').read_text(encoding='utf-8-sig')
script=original.replace("source=here/'hopper_expert_split_dependency.cu'","source=here/'device_only.cu'").replace('for variant in (2,):','for variant in (0,):').replace("f'wgs_split_dependency_{variant}_sm90a.cubin'","'pr50800_sm90a.cubin'").replace("f'-DREGISTER_DEQUANT={variant}','-DREGISTER_FINITE=1','-DSPLIT_READY=1',",'').replace("f'wgs_split_dependency_{variant}_compile.log'","'compile.log'").replace("f'wgs_split_dependency_{variant}_sass.txt'","'sass.txt'").replace("'wgs_split_dependency_compile.json'","'compile.json'").replace('finite_scan_required=True','upstream_unmodified_math=True')
(out/'compile.py').write_text(script)
print(json.dumps(manifest,indent=2))
