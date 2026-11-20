# DFlash projection arithmetic on SM86

The full-width operation multiplies a contiguous BF16 hidden-state matrix by the transpose of an 8192 × 40960 BF16 weight. Torch 2.14.1+cu130 selected classic cuBLAS and reported a native workspace budget of 8,519,680 bytes on the 20-SM RTX 3050 6GB Laptop GPU. The earlier 32 MiB-only search missed a full-width vendor plan at 256 rows. The default stock BF16 local projection differed at 1,302,399 out of 2,097,152 output positions there.

Calling the installed cuBLASLt full-shape heuristic with the actual native workspace recovered an algorithm that produced all the same BF16 output bits on the full and eight local operations. The comparison covered separate random weights and inputs and a mirrored 256-element cancellation test. Both dense output and parent-strided output passed. Twelve of 949 legal expanded configurations per layout passed all three datasets. These observations include every retained bit on this card and these inputs. CUDA's reproducibility promise applies to repeated invocations of the same operation under its stated conditions. It does not establish equality after changing dimensions, GPU generations, driver/library revisions, or reduction schedules. A kernel name or a matching algorithm ID would also leave the partial-sum order unverified. [NVIDIA cuBLAS reproducibility](https://docs.nvidia.com/cuda/cublas/#results-reproducibility) · [NVIDIA cuBLASLt algorithm check](https://docs.nvidia.com/cuda/cublas/#cublasltmatmulalgocheck)

## Runtime discovery

The standalone prototype uses the installed libtorch native workspace accessor and asks the runtime cuBLASLt full-operation heuristic for plans. Each plan must pass the vendor's local-layout check, workspace limit, and verification of the full result and every local slice against Torch. No application-provided algorithm IDs or configurations are embedded. The ABI is detected at runtime. CUDA installation layout and libraries are resolved at build time. The exact PyTorch build used separate default workspace rules on Ampere and Hopper. [PyTorch handle pool](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cuda/CublasHandlePool.cpp#L164-L169)

The first full-shape vendor plan passed all five tested widths and four independently changed inputs and weights per width. Aggregate paired median results follow. Times are microseconds. Each complete serial variant calculates eight local products on the same GPU and concatenates all output columns. The single-shard column covers one eighth of the output and excludes any transport. The two setup calibration examples were already allocated. The startup column includes discovery and all complete-output comparisons.

| Rows | Full native | Native control | Eight serial and concatenate | One shard | Startup ms | Persistent local workspace bytes |
|---:|---:|---:|---:|---:|---:|---:|
|8|5664|5653|5859|801|2763|196608|
|32|5631|5628|6794|921|38.6|32|
|128|6619|6638|7708|1041|43.2|16|
|256|13754|13653|13820|1830|81.8|64|
|753|34584|34617|34717|4471|199.6|96|

Serial complete work was slower or near parity at each size. Time for one shard cannot be treated as a serving speedup. There is no NCCL, no H200 measurement, and no DFlash target/draft acceptance measurement in this experiment. The data are in [runtime_report.json](runtime_report.json). The extended search is in [sweep_report.json](sweep_report.json).

## Serving integration

The scored parent keeps a replicated ordinary bias-free Linear on the active unquantized DFlash branch. Its project_target_hidden method invokes that layer before hidden_norm. Preserving the replicated weights costs the original 640 MiB per rank and keeps the native implementation callable for unregistered shapes. The previous column-sharded submission stores 80 MiB per rank and would require a separate replicated copy or a weight reconstruction to do the same fallback. Hooks must use the same draft tensor-parallel group and preserve output feature order before the norm.

[sglang_adapter.py](sglang_adapter.py) registers shapes collectively during startup after weight loading, caches one plan per local stream and shape, and uses native on an unknown shape. The prototype binds a plan to one stream. Additional serving streams require their own pre-registered plan and scratch. Any rank declining a plan makes the entire group use native for that shape. A local layout or stream rejection on a registered shape emits that rank's corresponding native slice into the usual gather to preserve collective order. This protocol passed a real two-process Gloo CPU test in [adapter_test.py](adapter_test.py). No eight-way CUDA transport or CUDA graph capture of this adapter was measured.

Strict mode also executes the original layer and returns its exact bits on any runtime divergence, including signed zeros and distinct NaN payloads. The empirical mode skips that full check and therefore has a different numerical assurance. The implementation is a standalone integration prototype. Binding the empirical mode as the default would not establish native-equivalent semantics on H200. The security of the fallback depends on identical logical batch shapes across the ordinary TP group, as required by the original draft projection collective.

## Reproduction

Within the active CUDA/PyTorch Linux environment, run the following commands from this directory. The audit uses the adjacent research directory's standalone layout helper. The runtime selector compiles independently.

~~~bash
python build.py sweep
python build.py runtime_lt
python audit_match.py sweep
python runtime_bench.py --out runtime_report.json
python fallback.py
python adapter_test.py
python full_width.py
~~~

