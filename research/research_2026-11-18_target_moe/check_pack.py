import argparse
import ast
import json
from pathlib import Path
import time
import torch
from hopper_pack import pack_experts
from qualify_packing import DONOR,pack_stage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    out = p.parse_args().out
    names = {"_wgs_copy","_pack_codes_nat","_stored_pos_mul"}
    n = [x for x in ast.parse(DONOR.read_text()).body if getattr(x,"name",None) in names]
    ns = {"torch":torch}
    exec(compile(ast.Module(n,[]),str(DONOR),"exec"),ns)
    torch.manual_seed(312)
    rows = []
    for n,stage in ((8192,"w13"),(4096,"w2")):
        w = torch.randint(256,(32,n,2048),device="cuda",dtype=torch.uint8)
        s = torch.randint(256,(32,n,256),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)
        result = pack_experts(w,s,stage=="w13").view(32,-1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        repeat = pack_experts(w,s,stage=="w13").view(32,-1)
        torch.cuda.synchronize()
        ms = (time.perf_counter()-start)*1000
        assert torch.equal(result,repeat)
        ran = []
        for e in (0,11,31):
            ref = pack_stage(w[e:e+1],s[e:e+1],stage,ns["_wgs_copy"],ns["_stored_pos_mul"]())
            assert torch.equal(result[e],ref.reshape(-1)),(stage,e)
            ran.append(e)
        rows.append(dict(stage=stage,bytes_oracle=ref.numel()*len(ran),experts_checked=ran,
                         total_bytes=result.numel(),warm_pack_ms=ms))
        del repeat,result,w,s,ref
    (out/"results.json").write_text(json.dumps(dict(device=torch.cuda.get_device_name(),rows=rows),indent=2))

if __name__=="__main__":
    main()
