"""Normalize PTX lane-selector and FP32 reduction trees by logical component."""
import hashlib
import json
import re
from pathlib import Path
here=Path(__file__).resolve().parent
rows=json.loads((here/"offline.json").read_text())
cache=Path.home()/".triton/cache"
def tree(row,full):
    stem=row["kernel"];cubin=next(p for p in cache.glob(f"*/{stem}.cubin") if hashlib.sha256(p.read_bytes()).hexdigest()==row["cubin_sha256"])
    text=cubin.with_suffix(".ptx").read_text()
    prefix=text.split("shfl.sync.bfly",1)[0]
    vec=re.findall(r"ld.global.v4.b32\s+\{([^}]+)\}",prefix)
    assert len(vec)==4,(row["sm"],stem,len(vec))
    vec=vec[:2] if full else vec[-2:]
    labels={};lanes={}
    for bank,regs in enumerate(vec):
        for j,k in enumerate(re.findall(r"%r\d+",regs)):lanes[k]=[bank*8+j*2,bank*8+j*2+1]
    for line in prefix.splitlines():
        u=re.search(r"mov.b32\s+\{\s*(%rs\d+),\s*(%rs\d+)\s*\},\s*(%r\d+)",line)
        if u and u[3] in lanes:
            labels[u[1]],labels[u[2]]=map(lambda j:("weight",j),lanes[u[3]])
        u=re.search(r"cvt.f32.bf16\s+(%r\d+),\s*(%rs\d+)",line)
        if u and u[2] in labels:labels[u[1]]=labels[u[2]]
        u=re.search(r"mul.f32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+)",line)
        if u:
            a,b=labels.get(u[2]),labels.get(u[3])
            if a and a==b and a[0]=="value": labels[u[1]]=("square",a[1])
            elif a and a[0]=="weight": labels[u[1]]=("value",a[1])
            elif b and b[0]=="weight": labels[u[1]]=("value",b[1])
        u=re.search(r"fma.rn.f32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+),\s*(%r\d+)",line)
        if u and u[2] in labels and u[2]==u[3] and u[4] in labels:
            labels[u[1]]=("fma",labels[u[2]][1],labels[u[4]])
        u=re.search(r"add.f32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+)",line)
        if u and u[2] in labels and u[3] in labels:
            labels[u[1]]=("add",*sorted((labels[u[2]],labels[u[3]]),key=repr))
    shuffles=re.findall(r"shfl.sync.bfly.b32\s+%r\d+,\s*(%r\d+),\s*(\d+),",text)
    length=8 if full else 7
    return dict(thread_tree=labels[shuffles[0][0]],mask_sequence=[int(v) for _,v in shuffles[:length]],cubin=row["cubin_sha256"])
tests=[]
for sm in (86,90):
    f=tree(next(r for r in rows if r["sm"]==sm and r["kernel"].startswith("_post_")),True)
    p=tree(next(r for r in rows if r["sm"]==sm and r["kernel"]=="_square"),False)
    t=dict(sm=sm,parent=f,shard=p,same_per_thread=f["thread_tree"]==p["thread_tree"],
        identical_lane_merge=f["mask_sequence"][:5]==p["mask_sequence"][:5],
        even_odd_warp_merge=f["mask_sequence"][5:]==[4,2,1] and p["mask_sequence"][5:]==[2,1])
    tests.append(t);print(sm,t["same_per_thread"],f["thread_tree"],p["thread_tree"],flush=True)
(here/"warp_tree.json").write_text(json.dumps(tests,indent=2))
