"""Execute real worker stream choreography on a deliberately adversarial CPU queue."""
import ast
from contextlib import contextmanager,nullcontext
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent/"decode_overlap"

class Queue:
    def __init__(self,name,events):
        self.name=name
        self.events=events
        self.tail=None
    def submit(self,op,fn=lambda:None,wait=None):
        index=len(self.events)
        self.events.append((op,self.name,[i for i in (self.tail,wait) if i is not None],fn))
        self.tail=index
    def wait_stream(self,other):
        self.submit("wait:"+other.name,wait=other.tail)

class Buffer:
    def __init__(self,storage):
        self.x=storage
        self.dtype=storage.dtype
        self.device="cuda:0"
        self.streams=[]
    @property
    def ndim(self):return self.x.ndim
    @property
    def shape(self):return self.x.shape
    def record_stream(self,s):self.streams.append(s.name)

class Cuda:
    def __init__(self,main):self.current=main;self.capture=False
    @contextmanager
    def stream(self,s):
        former=self.current;self.current=s
        try: yield
        finally:self.current=former
    def is_current_stream_capturing(self):return self.capture
    def current_stream(self,d):return self.current

def method():
    tree=ast.parse((ROOT/"worker.py").read_bytes())
    f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_project_during_verify")
    return ast.Module(body=[f],type_ignores=[])

def main():
    torch.set_num_threads(1);g=torch.Generator().manual_seed(71435)
    target=Buffer(torch.zeros(8,96,dtype=torch.bfloat16))
    fc=torch.randn(32,96,dtype=torch.bfloat16,generator=g)
    weight=torch.rand(32,generator=g)
    def project(h):
        p=F.linear(h,fc).float()
        return (p*torch.rsqrt(p.square().mean(-1,keepdim=True)+1.e-5)*weight).bfloat16()
    ctxs=0
    for cutoff in ("normal","verify_failure","project_failure"):
        for fallback in ("none","eager","nested-capture","shape","no-stream"):
            for forward_name in ("nondefault-forward-0","nondefault-forward-1"):
                for seed in (17,75,149,981):
                    gen=torch.Generator().manual_seed(seed)
                    events=[];forward=Queue(forward_name,events);side=Queue("context",events);cuda=Cuda(forward)
                    scope={"torch":NS(cuda=cuda,bfloat16=torch.bfloat16,inference_mode=nullcontext),
                           "contextmanager":contextmanager,"draft_tp_context":lambda x:nullcontext()}
                    exec(compile(method(),"actual-overlap-helper","exec"),scope)
                    launch=scope["_project_during_verify"]
                    def deferred_project(buffer):
                        result=Buffer(torch.empty(8,32,dtype=torch.bfloat16))
                        cuda.current.submit("FC+RMS",lambda:result.x.copy_(project(buffer.x)))
                        if cutoff=="project_failure":raise ArithmeticError("fixture")
                        return result
                    model=NS(fc=NS(in_features=96,weight=NS(device="cuda:0")),project_target_hidden=deferred_project)
                    worker=NS(_context_projection_stream=None if fallback=="no-stream" else side,
                              draft_parallel_layout=None,draft_model=model)
                    hrows=9 if fallback=="shape" else 8; active=fallback!="eager";cuda.capture=fallback=="nested-capture"
                    inp=torch.randn(8,96,dtype=torch.bfloat16,generator=gen)
                    got=[];target.streams=[]
                    forward.submit("target graph write",lambda:target.x.copy_(inp))
                    try:
                        with launch(worker,target,hrows,active) as out:
                            assert (out is None)==(fallback!="none")
                            forward.submit("TP all_gather")
                            if cutoff=="verify_failure":raise IndexError("fixture")
                            forward.submit("accept")
                            forward.submit("publish")
                        if out is not None:
                            assert out.streams==[forward_name]
                            forward.submit("KV writes",lambda out=out:got.append(out.x.clone()))
                        else:
                            forward.submit("native FC+RMS/KV",lambda:got.append(project(target.x)))
                    except (ArithmeticError,IndexError):
                        pass
                    forward.submit("overwrite graph aux",lambda:target.x.fill_(-999.))
                    # Prefer main work as long as ready; an unfenced side read races the overwrite.
                    done=set();trace=[]
                    while len(done)<len(events):
                        viable=[i for i,(op,q,deps,fn) in enumerate(events) if i not in done and all(x in done for x in deps)]
                        assert viable
                        i=min(viable,key=lambda i:events[i][1]!=forward_name)
                        op,q,_,fn=events[i];fn();done.add(i);trace.append((op,q))
                    if fallback=="none":
                        assert target.streams==["context"]
                        assert trace.index(("FC+RMS","context"))<trace.index(("overwrite graph aux",forward_name))
                    for op,q in trace:
                        if op in ("TP all_gather","accept","publish","KV writes"):assert q==forward_name
                    if got:assert torch.equal(got[0],project(inp))
                    ctxs+=1
    doc=dict(scope="Actual staged worker contextmanager with adversarial CPU stream/task executor, real BF16 FC+RMS numerical fixture; CUDA scheduling, allocator, TP network and timings absent.",
             scenarios=ctxs,forward_streams=2,exception_fences=True,changed_aux_safe=True,collectives_stay_on_forward=True)
    (ROOT/"ordering_cpu.json").write_text(json.dumps(doc,indent=2))
    print(doc)
if __name__=="__main__":main()
