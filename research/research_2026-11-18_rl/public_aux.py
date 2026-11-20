"""Fetch the standalone normalizations and the first TP-rank KV head for each draft layer."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from urllib.request import urlopen

from public_fc import HERE, REV, partial


def main():
    size=struct.unpack("<Q",partial(0,7))[0]
    header=json.loads(partial(8,size+7))
    config_url=f"https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/resolve/{REV}/config.json"
    with urlopen(config_url,timeout=30) as request:
        config=json.load(request)
    assert config["num_hidden_layers"]==5 and config["head_dim"]==128 and config["num_key_value_heads"]==8
    items=[("hidden_norm.weight",[8192])]
    for i in range(5):
        prefix=f"layers.{i}.self_attn."
        items.extend([(prefix+"k_proj.weight",[128,8192]),(prefix+"v_proj.weight",[128,8192]),
                      (prefix+"k_norm.weight",[128])])
    def fetch(item):
        key,shape=item
        record=header[key]
        native=record["shape"]
        assert native==shape or (native==[1024,8192] and shape==[128,8192])
        scalar_bytes={"BF16":2,"F16":2,"F32":4}[record["dtype"]]
        elements=1
        for dim in shape: elements*=dim
        first=8+size+record["data_offsets"][0]
        end=first+elements*scalar_bytes-1
        for attempt in range(3):
            try:
                payload=partial(first,end)
                break
            except OSError:
                if attempt==2:raise
        return payload,dict(shape=shape,dtype=record["dtype"],source_shape=native,first_byte=first,
            last_byte=end,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    tmp=Path(tempfile.gettempdir())/f"okc-dflash-aux-{REV[:12]}.raw"
    metadata={}
    with ThreadPoolExecutor(max_workers=3) as pool,tmp.open("wb") as fout:
        for (key,_),(payload,desc) in zip(items,pool.map(fetch,items)):
            desc["offset"]=fout.tell()
            fout.write(payload)
            metadata[key]=desc
    record=dict(temporary_file=str(tmp.resolve()),revision=REV,tensors=metadata,bytes=tmp.stat().st_size,
        sha256=hashlib.sha256(tmp.read_bytes()).hexdigest(),config_url=config_url,
        rope_theta=config["rope_theta"],rms_norm_eps=config["rms_norm_eps"],rope_parameters=config["rope_parameters"])
    (HERE/"public_aux_receipt.json").write_text(json.dumps(record,indent=2)+"\n")
    print("Read",len(items),"real tensors or TP slices",record["bytes"],"bytes",record["sha256"])


if __name__=="__main__":main()
