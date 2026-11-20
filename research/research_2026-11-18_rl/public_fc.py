"""Read just the BF16 context projection from the public DFlash2 safetensors checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
from urllib.request import Request, urlopen

HERE=Path(__file__).resolve().parent
REV="46c90f412549c472941a68382d91a99e5e6372a0"
URL=f"https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/resolve/{REV}/model.safetensors"


def partial(start,end):
    # A range-specific cache key allows HTTP caches to serve disjoint byte ranges safely.
    url=URL+f"?download=true&range_start={start}&range_end={end}"
    req=Request(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity",
                             "User-Agent":"open-kernel-local-benchmark/1"})
    response=urlopen(req,timeout=90)
    match=re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",response.headers.get("Content-Range",""))
    if response.status!=206 or match is None or (int(match[1]),int(match[2]))!=(start,end):
        response.close()
        raise IOError("server did not honor the requested byte range")
    with response:
        block=response.read(end-start+2)
    if len(block)!=end-start+1:
        raise IOError("short range response")
    return block


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fetch",action="store_true")
    args=p.parse_args()
    size=struct.unpack("<Q",partial(0,7))[0]
    if not 2<=size<=4_000_000:
        raise ValueError("unexpected safetensors header length")
    header=json.loads(partial(8,7+size))
    keys=[k for k,v in header.items() if k!="__metadata__" and v["shape"]==[8192,40960] and v["dtype"]=="BF16"]
    if len(keys)!=1: raise ValueError(keys)
    key=keys[0]
    a,b=header[key]["data_offsets"]
    assert b-a==8192*40960*2
    first=8+size+a
    state=dict(url=URL,revision=REV,tensor=key,shape=header[key]["shape"],dtype="BF16",bytes=b-a,
               first_byte=first,last_byte=8+size+b-1,header_bytes=size)
    if args.fetch:
        cache=Path(tempfile.gettempdir())/f"okc-dflash-fc-{REV[:12]}.bf16"
        offset=cache.stat().st_size if cache.exists() else 0
        if offset>state["bytes"]:raise ValueError("existing temporary file has invalid size")
        chunk=16*1024**2
        with cache.open("ab") as fout:
            while offset<state["bytes"]:
                end=min(offset+chunk,state["bytes"])
                for attempt in range(3):
                    try:
                        block=partial(first+offset,first+end-1)
                        break
                    except (OSError,TimeoutError):
                        if attempt==2: raise
                fout.write(block)
                fout.flush()
                offset=end
                if offset%(128*1024**2)==0: print(offset, "bytes",flush=True)
        h=hashlib.sha256()
        with cache.open("rb") as stream:
            for block in iter(lambda: stream.read(chunk),b""):
                h.update(block)
        state.update(sha256=h.hexdigest(),temporary_file=str(cache.resolve()))
    (HERE/"public_fc_receipt.json").write_text(json.dumps(state,indent=2)+"\n")
    print(json.dumps(state))


if __name__=="__main__": main()
