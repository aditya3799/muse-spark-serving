import json,pathlib,datetime,math
p=pathlib.Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19/coordination')
notes={
'LMCache/LMCache':('primary benchmark screened','April2026 10x headline is cross-DP shared400GB host KV forQwen235B FP8 on8H100/vLLM0.18.1. MeanTTFT13.6x and decode3.8x vs fragmented per-process offload. Not fixedTP8 Muse; lmcache launchflags fixed. Generic cachepressure insight only.'),
'InternLM/lmdeploy':('README hardware/model screened','TurboMind reports MXFP4 gpt-oss acceleration onH800; format/model differ fromMuse NVFP4 group16. PD/FlashMLA/DeepGEMM/DeepEP are not a direct source-only replacement.'),
'ModelTC/lightllm':('deployment documentation screened','DeepSeek H200 TP8 and DP8/EP recipes and microbatch overlap are documented. MLA/model differences and replicated-attention memory must be measured before a Muse transfer.'),
'huggingface/text-generation-inference':('README screened','Continuous batching, optimized attention and generic speculative decoding are already represented in pinnedSGLang. No retrieved exactMuse H2002x evidence.'),
'kvcache-ai/ktransformers':('hardware/workload mismatch','CPU expert offload and consumer/Ascend deployments target different memory topologies. No direct resident8H200 serving gain established.'),
'kvcache-ai/Mooncake':('cache-system README screened','KV transport, PD deployment and cache scheduling are useful infrastructure. Fixed single-node serving cannot inherit multi-node disaggregation gains.'),
'microsoft/DeepSpeed-MII':('README mechanism screened','FastGen DynamicSplitFuse combines chunked prefill and decoding, with batchedKV and continuousbatching. These mechanisms already exist in the pinned serving baseline; no novel Muse-specific patch established.'),
'microsoft/DeepSpeed':('README mechanism screened','Training/offload and inference optimization framework; README hardware coverage does not establish a Muse NVFP4/H200 kernel replacement.'),
'PaddlePaddle/FastDeploy':('resolved README pointer; screened','RootREADME is only a pointer toREADME_CN.md, now actually retrieved. Paddle engine supports PD/cachetransfer, MTP and several quantization formats; W4AFP8 would lower activationprecision here. No retrieved Muse/H200 exact kernel evidence.'),
'vectorch-ai/ScaleLLM':('README lineage screened','C++ runtime lineage used byxLLM, with customattention/CUDAgraphs/prefixcache/chunkedprefill/speculation. PublicmodelsLlama/Gemma/Phi do not establish this Muse path.'),
'efeslab/Nanoflow':('primary paper and source screened','Nano-batching overlaps heterogeneous resources; reported1.91x vsTRTLLM onolderLlama/Mixtral suite. Author confirms its asyncCPU scheduler is alreadySGLang default sincev0.4. Intra-GPU overlap still requires resourceinterference measurements.'),
'flexflow/flexflow-serve':('primary README screened','SpecInfer tree-verification with separate small models and hybridTP/PP; generic1.3-2x claims have different models and baselines. FrozenDFlash cannot be silently replaced.'),
'modular/modular':('official supported-model list screened','MAX supports broadBF16/FP8/FP4 architectures and DFlashKimi/Llama. Retrieved modeltable does not list Muse; compiler/runtimeport is not an inspectable drop-in H200 kernel gain.'),
'NVIDIA/hyperflare-inference':('primary architecture screened','Local SM103/120/121 prototype; PyTorchCUDA13.2 preparation exportsgraphs toRust. Muse recipeexists, but hardware/dependency/backend differ. Caller-ownedworkspace and completecontinuationstate ideas useful; noHopperclaim.'),
'aninrusimha/flashformer':('resolved repository move','README explicitly redirects tocheetah-lang/flashformer. Preserveoldretrieval aspointer, notimplementationevidence.'),
'cheetah-lang/flashformer':('primary README discovered','Whole-forwardkernel research built onCheetah; retrievedREADMEveryshort andnoqualifiedMuse implementation. Detailedsource audit remains open.'),
'mlsys2025/FlashDecoding++':('unverified guessed repository; paper found','Guessedrepo notvalidated. Primary MLSys2024 paper describes asynchronoussoftmax, flatGEMM andheuristics; olddense-modelcomparisons do not establish Muse FP8attention equivalence.'),
'NVIDIA/TensorRT-LLM':('leaf implementation audit','NVFP4W4A16 does supportSM80+, but tracedleaf dequantizesactiveexperts into fullBF16W13+W2 allocations (~3.22GB/rank/layer), and folds global scalebeforecoefficientrounding. Directport failsourmemory/arithmeticpremise. Boundedstaged andsharedpacked alternatives independentlyinvestigated.'),
'NVIDIA/cutlass':('active Hopper implementation','Exactjudgeheaders/API adapted; sharedHumminglayout devicecompile reached optimizer thenlocal5GiBVMlimit. NoHopperkernelbinary/runtime. Bounded packedrepack is a distinct fallbackdesign, pendingqualification.')}
s=(p/'build_source_audit.py').read_text()
s=s.replace("('focused_manifest.json','corrected_manifest.json')","('focused_manifest.json','corrected_manifest.json','late_manifest.json','followup_manifest.json')")
s=s.replace('out=[]\n','notes.update('+repr(notes)+')\nout=[]\n',1)
(p/'build_source_audit.py').write_text(s)
exec(compile(s,str(p/'build_source_audit.py'),'exec'),{'__file__':str(p/'build_source_audit.py')})
parent=1.2363;required=2/parent
out={'scope':'Amdahl necessary-condition illustration only. Component fractions on H200 are UNKNOWN; local speedup is not transferred or multiplied into the official score. Frontier normalization can differ from uniform speed scaling.','parent_official_score':parent,'uniform_parent_speedup_needed_for_2x':required,'assuming_component_speedup':[],'overlap_rules':['N-shard androw-shard optimize SAME DFlashFC; differentgeometrypolicies, nevermultiplytheirratios.','Cachepolicy affects TTFT/recomputation andmemorypressure; do not also attributeavoidedwork to kernels.','MoE mainloop andstaged/sharedpacked implementations arealternatives; benchmarkonecoherentdispatch policy.']}
for gain in (2,4.32,6,8,float('inf')):
 fraction=(1-1/required)/(1-(0 if math.isinf(gain) else 1/gain))
 out['assuming_component_speedup'].append({'component_gain':str(gain),'minimum_parent_time_fraction_for_uniform_2x_stock':fraction,'parent_gain_by_component_fraction':{str(f):1/(1-f+(0 if math.isinf(gain) else f/gain)) for f in (.05,.1,.2,.3,.4,.5)}})
(p/'component_budget.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out['assuming_component_speedup']))
