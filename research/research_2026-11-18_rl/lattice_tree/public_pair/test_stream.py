"""Check multi-tensor and tied-head streaming on a local synthetic safetensors file."""
import hashlib
import json
import struct
import torch
import stream_pair as loader
def one_chunk_case(names,data,chunk_size,duplicate=None):
    header={};offset=0
    for name,tensor in zip(names,data):
        n=tensor.numel()*2
        header[name]=dict(data_offsets=[offset,offset+n],dtype="BF16",shape=list(tensor.shape));offset+=n
    h=json.dumps(header).encode();raw=struct.pack("<Q",len(h))+h+b"".join(t.view(torch.uint8).numpy().tobytes() for t in data)
    loader.chunks=lambda *_args:((pos,raw[pos:pos+chunk_size]) for pos in range(0,len(raw),chunk_size))
    return loader.stream_file("","fixture",dict(size=len(raw),lfs={"oid":hashlib.sha256(raw).hexdigest()}),"cpu",header,duplicate)
a=torch.linspace(-1,1,600,dtype=torch.bfloat16);b=torch.arange(15).to(torch.bfloat16)
for n in (384,505,1024):
    values,digests=one_chunk_case(["model.embed_tokens.weight","b"],[a,b],n)
    assert torch.equal(values["model.embed_tokens.weight"],a) and torch.equal(values["b"],b)
    head,unused=one_chunk_case(["lm_head.weight"],[a],n,digests["model.embed_tokens.weight"])
    assert not head
    head,unused=one_chunk_case(["lm_head.weight"],[b],n,digests["model.embed_tokens.weight"])
    assert torch.equal(head["lm_head.weight"],b)
print("9 safetensors stream checks passed")
