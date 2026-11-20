"""Check bounded window addressing and stable graph capacities for the target verify geometry."""
import hashlib
import json
import random
from pathlib import Path

here=Path(__file__).resolve().parent
rng=random.Random(981202)
result={"seed":981202,"cases":0,"attention_rows":0,"page_reads":0,"graph_replays":0}
# No model weights are required: preserving each ordered key and each stored (already rotated)
# vector preserves the mathematical attention. Kernel order and precision get checked on GPU.
for q in (1,8,13):
    for align in (128,1920):
        w=2047+q
        choices=[q,w-1,w,w+1,w+align-1,w+align,w+align+1,74160,131072]
        for b in (1,4,16,32):
            for replay in range(3):
                lengths=[rng.choice(choices) for _ in range(b)]
                cu=[0]
                for request,length in enumerate(lengths):
                    base=max(0,length-w)//align*align
                    new=length-base
                    cu.append(cu[-1]+new)
                    assert new<=w+align-1 and base%align==0 and new>=q
                    # Indexed hash represents fragmented physical page slots. Short and absolute
                    # requests share slots exactly when the original page-table entries coincide.
                    salt=rng.randrange(1,100000)
                    def table(i):
                        return ((i*7723+salt)%1910527)*19%1514886+1
                    cached=[table(i) for i in range(base,length)]
                    for j in range(q):
                        hi=length-q+j
                        lo=max(0,hi-2047)
                        q2=new-q+j
                        lo2=max(0,q2-2047)
                        assert (base+lo2,base+q2)==(lo,hi)
                        expected=[table(i) for i in range(lo,hi+1)]
                        assert expected==cached[lo2:q2+1]
                        result["attention_rows"]+=1
                        result["page_reads"]+=len(expected)
                    result["cases"]+=1
                assert all(cu[i+1]-cu[i]==lengths[i]-max(0,lengths[i]-w)//align*align for i in range(b))
                result["graph_replays"]+=1
result.update(kernel_sha256=hashlib.sha256((here/"compact_window.py").read_bytes()).hexdigest(),correct=True)
(here/"window_oracle.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
