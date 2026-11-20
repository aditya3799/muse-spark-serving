"""Read the public RadixArk selector header and bounded BF16 table slices used for CPU qualification."""
import json,hashlib,sys,struct,tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent;D=H/"decode_selector";sys.path.insert(0,str(H.parent/"research_2026-11-18_rl"))
from public_fc import partial,REV,URL
size=struct.unpack("<Q",partial(0,7))[0];assert 2<size<4_000_000
header=json.loads(partial(8,7+size));tensor_names=[k for k in header if k!="__metadata__"]
interesting={k:header[k] for k in tensor_names if "selector" in k or "confidence" in k.lower() or "fore_d" in k.lower()}
keys={tail:next(k for k in interesting if k.endswith(tail)) for tail in ["predecessor_codebook","successor_codebook","hidden_projection.weight"]}
items=[]
for tail,k in keys.items():
    e=header[k];assert e["dtype"]=="BF16";shape=list(e["shape"])
    if "codebook" in k:assert shape==[202048,256];shape[0]=4096
    else:assert shape==[256,8192]
    count=2*shape[0]*shape[1];start=8+size+e["data_offsets"][0]
    items.append((k,start,count,shape))
dest=Path(tempfile.gettempdir())/f"okc-selector-slices-{REV[:12]}.bf16"
def fetch(entry):
    k,start,count,shape=entry
    for attempt in range(3):
        try:return partial(start,start+count-1)
        except OSError:
            if attempt==2:raise
desc={}
with dest.open("wb") as o,ThreadPoolExecutor(max_workers=3) as pool:
    for (k,start,n,shape),buf in zip(items,pool.map(fetch,items)):
        desc[k]=dict(offset=o.tell(),length=n,shape=shape,source_shape=header[k]["shape"],sha256=hashlib.sha256(buf).hexdigest())
        o.write(buf)
rec=dict(revision=REV,url=URL,tensor_count=len(tensor_names),all_tensor_names=tensor_names,file=str(dest),bytes=dest.stat().st_size,header_sha256=hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest(),slices=desc)
(D/"real_selector_weights.json").write_text(json.dumps(rec,indent=2));print(dest.stat().st_size,keys,list(interesting))

