"""Test the same placement policy on chronologically disjoint public top-eight routing traces."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import URLError
import numpy as np
from placement import NATIVE, select, route_counts, peak


HERE=Path(__file__).resolve().parent
HF="https://huggingface.co/datasets/UW-SyFI/servingstudio-workload"
REV="9c64b393b4d5769ffb88d5d74a1f67f6985dea4f"
CAPTURES={
    "glm53":"glm53_nvfp4/vllm_dflash2_k7/diverse_100/capture/20260922",
    "glm52":"glm52_nvfp4/vllm/diverse_100/capture/20260922",
}
LAYERS=[0,9,19,29,39,49,59,69,74]
STARTS=[0,57344,110592]
COUNT=8192


def read(url,headers=None):
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers=headers or {}),timeout=45) as r:
                if headers and not r.headers.get("Content-Range","").startswith("bytes "+headers["Range"][6:]+"/"):
                    raise ValueError(f"Server did not serve requested range: {r.headers.get('Content-Range')}")
                return r.read()
        except (OSError,URLError):
            if attempt==2:raise


def fetch(name,folder):
    base=f"{HF}/resolve/{REV}/{folder}"
    config=json.loads(read(base+"/manifest.json"))
    assert config["num_experts"]==256 and config["top_k"]==8 and config["num_layers"]==75
    prov=json.loads(read(base+"/provenance.json"))
    stride=config["num_layers"]*8*2
    arrays=[]
    chunks=[]
    for row in STARTS:
        first,last=row*stride,(row+COUNT)*stride-1
        # Make the response cache key specific to each byte range.
        data=read(base+f"/routes.u16?start={first}",{"Range":f"bytes={first}-{last}"})
        assert len(data)==COUNT*stride
        routes=np.frombuffer(data,dtype="<u2").reshape(COUNT,75,8)[:,LAYERS,:].copy()
        assert routes.max()<256
        arrays.append(routes)
        chunks.append(dict(token_start=row,byte_start=first,byte_end=last,sha256=hashlib.sha256(data).hexdigest()))
    path=HERE/(name+"_routes.npz")
    np.savez_compressed(path,arr=np.stack(arrays))
    receipt=dict(repo=HF,revision=REV,capture=folder,manifest=config,selected_layers=LAYERS,chunks=chunks,
        model=prov.get("model"),workload=prov.get("workload"),local_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (HERE/(name+"_source.json")).write_text(json.dumps(receipt,indent=2))


def metrics(a):
    return dict(median=float(np.median(a)),p10=float(np.percentile(a,10)),p90=float(np.percentile(a,90)),min=float(np.min(a)),max=float(np.max(a)))


def main():
    summary={}
    for name,folder in CAPTURES.items():
        path=HERE/(name+"_routes.npz")
        if not path.exists():fetch(name,folder)
        receipt=json.loads((HERE/(name+"_source.json")).read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt["local_sha256"]
        arr=np.load(path)["arr"]
        rows={}
        for m in (128,768):
            for region in range(len(STARTS)):
                for lid,layer in enumerate(LAYERS):
                    training=arr[region,:2048,lid].astype(np.int32)
                    mapping,actions=select(route_counts(training),4)
                    for agreement in ("same_segment","distant_segment"):
                        dest=region if agreement=="same_segment" else (region+1)%len(STARTS)
                        gains=[]
                        for start in range(2048,COUNT-m+1,m):
                            counts=route_counts(arr[dest,start:start+m,lid])
                            gains.append(peak(counts,NATIVE,16)/peak(counts,mapping,16))
                        rows[f"M{m}_region{region}_layer{layer}_{agreement}"]=dict(ratios=gains,swaps=len(actions))
        summary[name]={}
        for m in (128,768):
            for agreement in ("same_segment","distant_segment"):
                values=[v for k,v0 in rows.items() if k.startswith(f"M{m}_") and k.endswith("_"+agreement) for v in v0["ratios"]]
                summary[name][f"M{m}_{agreement}"]=dict(n=len(values),**metrics(values))
        (HERE/(name+"_results.json")).write_text(json.dumps(rows))
    print(json.dumps(summary))
    (HERE/"real_route_summary.json").write_text(json.dumps(dict(scope="Measured expert assignments in other models; counts of 16-row expert tiles on busiest logical rank only. Held-out rows exclude training.",results=summary),indent=2))

if __name__=="__main__":main()
