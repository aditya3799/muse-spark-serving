"""Stream public safetensors directly into bounded device buffers, checking complete LFS hashes."""
import hashlib
import json
import os
import struct
import time
import urllib.request
from pathlib import Path

import torch
from inspect_pair import TARGET, DRAFT
HERE = Path(__file__).resolve().parent
MIB = 1024*1024
def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()
def files(base):
    repo, revision = base.split("https://huggingface.co/")[1].split("/resolve/")
    api = f"https://huggingface.co/api/models/{repo}/tree/{revision.strip('/')}?expand=true"
    return {r["path"]:r for r in json.loads(fetch(api)) if "lfs" in r}

def read_part(url,start,end):
    for retry in range(4):
        try:
            params=f"?download=true&offset={start}&attempt={retry}"
            request=urllib.request.Request(url+params,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"})
            with urllib.request.urlopen(request,timeout=70) as resp:
                if resp.status!=206 or not resp.headers.get("Content-Range","").startswith(f"bytes {start}-{end}/"):
                    raise ValueError("range response differs from request")
                payload=resp.read(end-start+2)
            if len(payload)!=end-start+1:raise IOError("incomplete public file range")
            return payload
        except (OSError,ValueError):
            if retry==3:raise
            time.sleep(1+2*retry)

def chunks(url,size,unit=32*MIB):
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    offsets=iter(range(0,size,unit))
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending=deque()
        def submit():
            start=next(offsets,None)
            if start is not None:
                pending.append((start,pool.submit(read_part,url,start,min(size,start+unit)-1)))
        for _ in range(3):submit()
        while pending:
            start,task=pending.popleft()
            payload=task.result()
            submit()
            yield start,payload

def stream_file(base,filename,meta,device,header,dupe_hash=None):
    total=meta["size"]
    store={}; digests={}; sha=hashlib.sha256()
    tensors=sorted((info["data_offsets"][0],info["data_offsets"][1],name,info["shape"]) for name,info in header.items() if name!="__metadata__")
    assert all(info["dtype"]=="BF16" for name,info in header.items() if name!="__metadata__")
    hbytes=None;current=0;duplicate=dupe_hash is not None
    single_hash=hashlib.sha256() if duplicate else None
    for start,payload in chunks(base+filename,total):
        sha.update(payload)
        if start==0:
            hbytes=8+struct.unpack("<Q",payload[:8])[0]
            assert json.loads(payload[8:hbytes])==header
        lo=start-hbytes;hi=lo+len(payload)
        while current<len(tensors):
            beg,end,name,shape=tensors[current]
            if hi<=beg:break
            a=max(lo,beg);b=min(hi,end)
            if name not in store and not duplicate:
                store[name]=torch.empty(shape,dtype=torch.bfloat16,device=device)
            piece=memoryview(payload)[a-lo:b-lo]
            if duplicate:
                single_hash.update(piece)
            else:
                if name=="model.embed_tokens.weight":
                    digests.setdefault(name,hashlib.sha256()).update(piece)
                # Safetensors padding and partial network chunks need not be aligned for a BF16 view.
                raw=torch.frombuffer(bytearray(piece),dtype=torch.uint8)
                store[name].view(-1).view(torch.uint8)[a-beg:b-beg].copy_(raw)
            if b<end:break
            current+=1
        if start%(256*MIB)==0:print(filename,round(100*(start+len(payload))/total,1),"%",flush=True)
    assert current==len(tensors) and sha.hexdigest()==meta["lfs"]["oid"]
    if duplicate:
        if single_hash.hexdigest()!=dupe_hash:
            return stream_file(base,filename,meta,device,header)
    return store,{k:v.hexdigest() for k,v in digests.items()}

def state(device):
    headers=json.loads((HERE/"headers.json").read_text())
    target_meta=files(TARGET);draft_meta=files(DRAFT)
    first,second=sorted(headers["target"])
    target,hashes=stream_file(TARGET,first,target_meta[first],device,headers["target"][first])
    second_state,_=stream_file(TARGET,second,target_meta[second],device,headers["target"][second],hashes["model.embed_tokens.weight"])
    target["lm_head.weight"]=second_state.get("lm_head.weight",target["model.embed_tokens.weight"])
    print("shared embedding and LM head",not bool(second_state),flush=True)
    draft,_=stream_file(DRAFT,"model.safetensors",draft_meta["model.safetensors"],device,headers["draft"]["model.safetensors"])
    return target,draft