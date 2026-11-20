"""Correctness-only test of an outside stream consuming a freshly signalled in-graph tensor."""
import ast,contextlib,faulthandler,json,time
from pathlib import Path
from types import SimpleNamespace as S
import torch

HERE=Path(__file__).resolve().parent/"decode_overlap"

def main():
    faulthandler.dump_traceback_later(30,repeat=True)
    n=0;outlog=[]
    def log(stage):
        entry=dict(event=stage,elapsed=round(time.monotonic()-start,3))
        outlog.append(entry)
        (HERE/"event_smoke_progress.json").write_text(json.dumps(outlog,indent=2))
        print(stage,flush=True)
    start=time.monotonic();log("start")
    fn=next(x for x in ast.walk(ast.parse((HERE/"event_dflash_worker_v2.py").read_bytes())) if isinstance(x,ast.FunctionDef) and x.name=="_project_during_verify")
    scope=dict(torch=torch,contextmanager=contextlib.contextmanager,draft_tp_context=lambda _:contextlib.nullcontext())
    exec(compile(ast.Module([fn],[]),"production-early-marker","exec"),scope)
    f=scope[fn.name];inputs={};forward=torch.cuda.Stream();side=torch.cuda.Stream()
    with torch.cuda.stream(forward):
        # A delayed checked write tests whether last replay's event could be observed.
        for rows in (4,8):
            h=torch.zeros(rows,64,dtype=torch.bfloat16,device="cuda")
            x=torch.zeros_like(h)
            e=torch.cuda.Event(external=True);graph=torch.cuda.CUDAGraph()
            y=torch.empty_like(h)
            with torch.cuda.graph(graph):
                torch.cuda._sleep(250000)
                h.copy_(x)
                e.record()
                torch.cuda._sleep(200000)
                torch.mul(x,3,out=y)
            inputs[rows]=(x,h,e,graph,y)
            log("captured_"+str(rows))
        worker=S(_context_projection_stream=side,draft_parallel_layout=None,
                 draft_model=S(fc=S(in_features=64,weight=S(device=torch.device("cuda",torch.cuda.current_device()))),
                               project_target_hidden=lambda z:z[:,:16]*2))
        expected=[];actual=[]
        for i in range(20):
            rows=(4,8)[i%2];x,h,e,graph,y=inputs[rows];x.fill_(i+1);graph.replay()
            # Every fourth request's returned hidden comes from an eager tail that
            # is unavailable at the graph marker; a full forward fence is required.
            copied=i%4==3
            live=(y.clone() if copied else h)[:rows-1]
            with f(worker,live,rows-1,True,(e,h)) as ctx:
                torch.cuda._sleep(50000)
            actual.append(ctx.clone())
            expected.append((i+1)*(6 if copied else 2))
            h.fill_(-998);n+=1
        forward.synchronize()
        for t,v in zip(actual,expected):assert torch.all(t==v).item(),(v,t[0,0].item())
    result=dict(scope="Actual two replayed CUDA graphs, external events and packaged worker stream fencing on small changed BF16 tensors; correctness-only and no timing estimate.",
                external_marker=True,steps=n,all_expected=True,different_graph_keys=2,eager_tail_copy_fallbacks=5)
    (HERE/"event_cuda_smoke.json").write_text(json.dumps(result,indent=2));log("verified")
if __name__=="__main__":
    with torch.inference_mode():main()
