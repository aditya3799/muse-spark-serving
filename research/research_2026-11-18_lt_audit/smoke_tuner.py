"""Exercise the complete-work runtime tuner through its normal entry point and graph replay."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from full_projector import CompleteProjection
from audit_match import K,N,BASELINE,mismatches
from runtime_projection import native_workspace
HERE=Path(__file__).resolve().parent

def main(args):
    generator=torch.Generator(device="cuda").manual_seed(args.seed)
    layer=torch.nn.Linear(K,N,bias=False,dtype=torch.bfloat16,device="cuda").eval()
    layer.weight.normal_(std=K**-.5,generator=generator)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    runner=CompleteProjection(layer,empirical=True)
    result={"seed":args.seed,"scope":"runtime-selected full-width FC on SM86, empirical mode",
            "native_workspace":native_workspace(),"baseline_sha256":BASELINE,"shapes":{}}
    with torch.cuda.stream(stream):
        for m in args.rows:
            def value(): return torch.empty(m,K,device="cuda",dtype=torch.bfloat16).normal_(generator=generator)
            train,check=[value()], [value(),value()]
            start=time.perf_counter()
            data=runner.prepare(train,check,pairs=4)
            torch.cuda.synchronize()
            data["wall_setup_seconds"]=time.perf_counter()-start
            data["eager_heldout_bad"]=[]
            x=value()
            answer=runner(x)
            data["eager_heldout_bad"].append(mismatches(answer,layer(x)))
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                static=runner(x)
            data["graph_replay_bad"]=[]
            data["successive_graph_output_changes"]=[]
            last=static.clone()
            second=torch.cuda.Stream()
            data["prepared_stream_is_nondefault"]=(stream.cuda_stream!=torch.cuda.default_stream().cuda_stream)
            data["separate_replay_stream"]=(second.cuda_stream!=stream.cuda_stream)
            for trial in range(4):
                replay_stream=second if trial%2 else stream
                replay_stream.wait_stream(stream)
                with torch.cuda.stream(replay_stream):
                    x.copy_(value())
                    graph.replay()
                    data["graph_replay_bad"].append(mismatches(static,layer(x)))
                    data["successive_graph_output_changes"].append(mismatches(static,last))
                    last.copy_(static)
                stream.wait_stream(replay_stream)
            with torch.cuda.stream(second):
                second.wait_stream(stream)
                data["unregistered_other_stream_is_fallback"]=(runner.key(x) not in runner.entries)
                data["other_stream_fallback_bad"]=mismatches(runner(x),layer(x))
            stream.wait_stream(second)
            functions={"baseline":layer,"control":layer,"candidate":runner}
            times={name:[] for name in functions}
            orders=[]
            for pair in range(6):
                x.copy_(value())
                order=list(functions)
                random.Random(args.seed+pair).shuffle(order)
                for visit in (order,list(reversed(order))):
                    orders.append(visit)
                    for name in visit:
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record()
                        y=functions[name](x)
                        b.record()
                        b.synchronize()
                        times[name].append(a.elapsed_time(b)*1000)
            data.update(hot_us=times,orders=orders)
            result["shapes"][m]=data
            args.out.write_text(json.dumps(result,indent=2)+"\n")
            print(m,"admitted",data.get("admitted"),"plans",data.get("vendor_plans"),"setup",data["wall_setup_seconds"],
                  "graph_bad",data["graph_replay_bad"],flush=True)
            del graph,static
    torch.cuda.synchronize()
    runner.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows",nargs="+",type=int,default=[8,32,128,256,753])
    p.add_argument("--seed",type=int,default=543653)
    p.add_argument("--out",type=Path,default=HERE/"anchor_report.json")
    with torch.inference_mode(): main(p.parse_args())

