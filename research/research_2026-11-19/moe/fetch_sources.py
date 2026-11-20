import json,hashlib,urllib.request,datetime,time
from pathlib import Path
base=Path(__file__).resolve().parent
out=base/'sources';out.mkdir(exist_ok=True)
requests=[('vllm-project/humming','main',['README.md','humming/include/humming/memory/g2s_loader/loader_bs.cuh','humming/include/humming/memory/g2s_loader/loader_b.cuh','humming/include/humming/kernel/humming.cuh','humming/include/humming/memory/g2s_pipeline.cuh']),('flashinfer-ai/flashinfer','main',['docs/design_docs/moe_ep_architecture.md','docs/design_docs/moe_ep_runbook.md','flashinfer/gemm/bf16_fp4.py']),('NVIDIA/TensorRT-LLM','main',['docs/source/developer-guide/moe-perf-guide.md']),('deepseek-ai/DeepGEMM','main',['README.md']),('deepseek-ai/DeepEP','main',['README.md'])]
receipts=[]
def get(url):
    r=urllib.request.Request(url,headers={'User-Agent':'okc-serving-research/1.0'})
    with urllib.request.urlopen(r,timeout=25) as f:
        data=f.read(12000000)
        if len(data)>=12000000:raise ValueError('source exceeds budget')
        return data
for repo,branch,paths in requests:
    rec={'repo':repo,'requested_branch':branch,'read_utc':datetime.datetime.utcnow().isoformat()+'Z','files':[]}
    try:
        c=json.loads(get('https://api.github.com/repos/'+repo+'/commits/'+branch));sha=c['sha'];rec['sha']=sha;rec['commit_date']=c['commit']['committer']['date']
        for p in paths:
            item={'path':p,'url':'https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+p}
            try:
                data=get(item['url']);name=repo.replace('/','__')+'__'+p.replace('/','__');(out/name).write_bytes(data)
                item.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),local=name)
            except Exception as e:item['error']=str(e)
            rec['files'].append(item)
    except Exception as e:rec['error']=str(e)
    receipts.append(rec);(base/'sources.json').write_text(json.dumps(receipts,indent=2));print(repo,rec.get('sha'),rec.get('error'),flush=True)
