"""Inspect public checkpoint manifests and pinned reference source. Transfer headers and text only."""
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

here = Path(__file__).resolve().parent
TARGET = "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/"
DRAFT = "https://huggingface.co/rasyosef/Qwen3-1.7B-DFlash2/resolve/c8b13ac1d5013995cdb9f80f362efc9c03319ec3/"
def get(url,limits=256000):
    with urllib.request.urlopen(url,timeout=40) as response:
        blob=response.read(limits+1)
        if len(blob)>limits:raise ValueError("unexpected reference size")
        return blob
def header(url):
    request=urllib.request.Request(url+"?download=true",headers={"Range":"bytes=0-131071","Accept-Encoding":"identity"})
    with urllib.request.urlopen(request,timeout=45) as response:
        size=struct.unpack("<Q",response.read(8))[0]
        if size>120000:raise ValueError("unexpected header size")
        return json.loads(response.read(size))
def main():
    report={}
    for name,url in (("target",TARGET),("draft",DRAFT)):
        config=json.loads(get(url+"config.json"))
        (here/(name+"_config.json")).write_text(json.dumps(config))
        files=sorted(set(json.loads(get(url+"model.safetensors.index.json"))["weight_map"].values())) if name=="target" else ["model.safetensors"]
        report[name]={f:header(url+f) for f in files}
        print(name,[(f,len(h)) for f,h in report[name].items()],flush=True)
        (here/"headers.json").write_text(json.dumps(report))
    revision=json.loads(get("https://api.github.com/repos/vllm-project/speculators/commits/main",900000))["sha"]
    refs={}
    for folder,fil in (("dflash2","model_definitions"),("dflash2","core"),("dflash","model_definitions"),("dflash","core")):
        url=f"https://raw.githubusercontent.com/vllm-project/speculators/{revision}/src/speculators/models/{folder}/{fil}.py"
        raw=get(url,300000)
        out=here/(folder+"_"+fil+".ref.txt")
        out.write_bytes(raw)
        refs[out.name]=dict(url=url,sha256=hashlib.sha256(raw).hexdigest())
    (here/"sources.json").write_text(json.dumps(refs,indent=2))
if __name__=="__main__":main()