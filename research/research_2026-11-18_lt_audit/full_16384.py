"""One bounded 16k-row complete-work allocation and a single timed call per arm."""
import json
import time
from pathlib import Path
import torch
from torch.nn.functional import linear
from full_width import data,timing,HOLDOUT
from audit_match import describe,load_lib,mismatches,K,N
from runtime_projection import native_workspace
HERE=Path(__file__).resolve().parent
M=16384
def main():
    free,total=torch.cuda.mem_get_info()
    record={"rows":M,"seed":HOLDOUT+190,"free_bytes":free,"total_bytes":total,"timings_per_arm":1}
    dst=HERE/"full_16384.json"
    if free<3.3*(2**30):
        record["reason"]="available memory below predeclared 3.3 GiB requirement"
        dst.write_text(json.dumps(record,indent=2))
        return
    prev=json.loads((HERE/"full_prefill_detail.json").read_text())["3072"]["chosen"]
    lib=load_lib()
    x,w,ref=data(M,HOLDOUT+190)
    budget=native_workspace()[1]
    scratch=torch.empty(32*2**20,device="cuda",dtype=torch.uint8)
    q=lib.make_sweep(M,N,N,K,N,budget,scratch.numel(),8)
    fields="id tile split reduction swizzle custom stages inner cluster".split()
    try:
        choices=[describe(lib,q,i) for i in range(lib.sweep_count(q))]
        i=next((i for i,row in enumerate(choices) if all(row[f]==prev[f] for f in fields)),None)
        record["transferred_3072"]=i is not None
        if i is None:
            i=0
            record["scope"]="3072 plan unavailable; current full-shape first vendor candidate"
        else: record["scope"]="3072 discovery descriptor checked as legal at 16384 rows"
        record["algorithm"]=choices[i]
        def candidate():
            out=torch.empty_like(ref)
            rc=lib.sweep_call(q,i,1,w.data_ptr(),x.data_ptr(),out.data_ptr(),scratch.data_ptr(),
                              scratch.numel(),torch.cuda.current_stream().cuda_stream)
            if rc: raise RuntimeError(str(rc))
            return out
        record["baseline_us"]=timing(lambda:linear(x,w))
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        answer=candidate()
        b.record()
        b.synchronize()
        record["candidate_us"]=a.elapsed_time(b)*1000
        record["outputs"]=answer.numel()
        record["mismatches"]=mismatches(answer,ref)
        record["peak_bytes"]=torch.cuda.max_memory_allocated()
        dst.write_text(json.dumps(record,indent=2)+"\n")
        print(record,flush=True)
    finally: lib.drop_sweep(q)
if __name__=="__main__": main()

