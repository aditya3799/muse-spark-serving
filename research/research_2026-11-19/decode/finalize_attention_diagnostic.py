"""Freeze source-only attention follow-up; no qualification or GPU execution."""
import ast,hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
sources={}
for name in ['build_attention_precision.py','precision_attention.py','precision_attention_manifest.json',
 'prepare_attention_diagnostic.py','attention_diagnostic_native.cu','attention_diagnostic_source.json',
 'hopper_attention_diagnostic.py','precision_attention_request_compile.json','precision_attention_flat_compile.json',
 'compile_attention_diagnostic.py','attention_diagnostic_compile.json','attention_diagnostic_initial_failure.json',
 'public_attention_rounding_oracle.py','public_attention_rounding_oracle.json',
 'attention_softmax_oracle.py','attention_softmax_oracle.json','fp8_conversion_audit.py','fp8_conversion_audit.json']:
 p=here/name
 if not p.exists():raise FileNotFoundError(p)
 if p.suffix=='.py':ast.parse(p.read_text())
 sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
doc=dict(scope=__doc__,sources=sources,python_syntax_passed=True,serving_qualified=False,
 local_kernel_execution=False,hopper_kernel_execution=False,diagnostic_abi_compiled=True,
 diagnostic_abi_cpu_layout_passed=True,diagnostic_library_sha256='8620e11ee2b345155d6f80da802c5128a59a5bf8c97367d4cd4fa478651618a7',
 donor_pin='6cae49a639ee6d14b929a5fd4a63c627f96d7dc0',
 donor_url='https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_attention_decode.py',
 fa3_pin='f89bc2306632d1ec5f97b014dded4254f5b4a907',
 fa3_pin_basis='Frozen SGLang297e54fd105e974cdf401e5b4d20253a22c16f90 AOT CMakeLists.txt pins sgl-project/sgl-attn archive f89bc; actual deployed binary hash still needs verification.',
 native_contract=['FA3 path uses BF16 Q/K normalization with kv_store=None; TRT fused FP8 KV-store path is not used.',
 'FA3 casts Q to FP8; pool separately casts BF16K/V toFP8, then native store_cache skips reserved slot0.',
 'Original FP32RoPE cos/sin cache+positions; BF16 normalized K and BF16 rotated K boundaries restored.',
 'Original softmaxscale=head_dim**-0.5*5.4730064863875; native FP8 softmax Max_offset8; FP32 probability denominator.',
 'BF16 attention output before original perheadRMS+sigmoidgate.',
 'Installed Torch2.14.1 CPU conversion/header saturate FP8finiteoverflow, but deployed CUDA exceptional-value behavior untested.'],
 remaining_gates=['Execute diagnostic on explicitly allocatedHopper; CABI build and CPU layoutchecks do not establish GPU correctness.',
 'Bytecompare complete updated K/V and reservedslot, output error against native FA3, allpadding/mixedcontexts/splits, repeated and CUDAgraph runs.',
 'Resolve reducedtile64vsnative128, forward-vs-reverse traversal and atomicmerge ordering effects; no acceptance threshold invented here.',
 'Validate race/memory/synchronization, exact backendconfig and actualcheckpoint values; no full model altered.',
 'Charge outputzeroing/staging/synchronization and measure entireattention chain and actualservingfrontier onlyaftercorrectness.'])
(here/'ATTENTION_DIAGNOSTIC_REVIEW.json').write_text(json.dumps(doc,indent=2))
print('syntax passed; source-only packet frozen; serving_qualified=False')
