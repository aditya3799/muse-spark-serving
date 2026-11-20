import concurrent.futures, datetime, hashlib, json, pathlib, subprocess, time, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'engine_sources'; OUT.mkdir(exist_ok=True)
REPOS={
'sgl-project/sglang':'engine', 'vllm-project/vllm':'engine', 'NVIDIA/TensorRT-LLM':'engine',
'NVIDIA/dynamo':'serving orchestration','InternLM/lmdeploy':'engine','ModelTC/lightllm':'engine',
'PaddlePaddle/FastDeploy':'engine','huggingface/text-generation-inference':'engine',
'ggml-org/llama.cpp':'engine','mlc-ai/mlc-llm':'engine','zml/zml':'engine',
'SyFI-ServingStudio/SSEngine-GLM5.3-Flash':'engine','facebookresearch/sgl-muse-mega':'model megakernel',
'deepseek-ai/FlashMLA':'attention','flashinfer-ai/flashinfer':'kernel library','Dao-AILab/flash-attention':'attention',
'deepseek-ai/DeepGEMM':'GEMM library','deepseek-ai/DeepEP':'expert communication','NVIDIA/cutlass':'GEMM library',
'Infini-AI-Lab/Satori':'engine','NVIDIA/Megatron-LM':'training and inference',
'Princeton-NLP/SpecInfer':'speculation','nvidia/cuda-quantization':'quantization',
'flashinfer-ai/flashinfer-bench':'kernel benchmarking','kvcache-ai/ktransformers':'heterogeneous engine',
'microsoft/DeepSpeed-MII':'engine','microsoft/DeepSpeed':'engine library',
'mozilla-ai/llamafile':'engine packaging','vllm-project/aibrix':'serving orchestration',
'kvcache-ai/Mooncake':'KV and disaggregation','LMCache/LMCache':'KV cache','tencent/Weaver':'engine',
'luongthecong123/smallpond':'unknown candidate','GeeeekExplorer/nano-vllm':'minimal engine',
'ztxz16/fastllm':'engine','AutoGPTQ/AutoGPTQ':'quantization legacy','turboderp-org/exllamav3':'engine',
'ROCm/aiter':'AMD kernel library','sgl-project/sgl-kernel':'kernel library',
'HazyResearch/ThunderKittens':'kernel library','HazyResearch/Megakernels':'persistent kernels',
'ucb-bar/megakernels':'persistent kernels','dangxingyu/megakernels':'persistent kernels',
'Red-Weasel/machx-inference-engine':'Intel engine','uv-genai/gufo-tp':'AMD engine',
'FluffyAIcode/Kakeya-LLM-Inference-engine':'research engine','antgroup/AntLLM':'engine',
'chenweioh/vortex':'engine candidate','flashinfer-ai/MegaMoE':'expert kernels',
}
def one(item):
 repo,kind=item; start=time.monotonic(); rec={'repository':repo,'kind':kind,'url':'https://github.com/'+repo,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  p=subprocess.run(['git','ls-remote','https://github.com/'+repo+'.git','HEAD'],capture_output=True,text=True,timeout=35)
  if p.returncode: raise RuntimeError(p.stderr.strip()[-400:])
  sha=p.stdout.split()[0]; rec['head_commit']=sha
  for name in ('README.md','README.rst','README'):
   url='https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+name
   try:
    req=urllib.request.Request(url,headers={'User-Agent':'OpenKernelChallenge-research/1.0'})
    with urllib.request.urlopen(req,timeout=25) as r: data=r.read(200001)
    if len(data)>200000: raise RuntimeError('README exceeds per-file budget')
    path=OUT/(repo.replace('/','__')+'__'+name.replace('.','_')+'.txt'); path.write_bytes(data)
    rec.update(readme_url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),local_path=str(path.relative_to(ROOT)),status='retrieved')
    txt=data.decode('utf-8','replace'); terms=('H200','Hopper','FP8','FP4','NVFP4','Muse','decode','prefill','speculat','DFlash','MoE','expert','continuous','graph','tensor parallel','cache')
    rec['matching_lines']=[{'line':i+1,'text':line[:500]} for i,line in enumerate(txt.splitlines()) if any(t.lower() in line.lower() for t in terms)][:65]
    break
   except urllib.error.HTTPError as e:
    if e.code!=404: raise
  else: rec['status']='no_readme'
 except Exception as e:rec.update(status='unavailable',error=str(e)[:500])
 rec['elapsed_s']=round(time.monotonic()-start,2); print(repo,rec['status'],rec.get('head_commit',''),flush=True); return rec
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 records=list(pool.map(one,REPOS.items()))
manifest={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Broad primary-source discovery; readme claims are not experimental confirmation. Frozen commit per retrieved project. Unknown candidates and inaccessible projects are retained, not inferred to exist.','budgets':{'projects':len(REPOS),'concurrent_fetches':4,'max_readme_bytes':200000,'git_timeout_s':35,'http_timeout_s':25},'records':records}
(ROOT/'engine_manifest.json').write_text(json.dumps(manifest,indent=2))
print('COMPLETE',len(records),'retrieved',sum(r['status']=='retrieved' for r in records),flush=True)
