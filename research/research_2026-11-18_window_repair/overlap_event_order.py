"""Exercise early external graph marker and eager/post-copy fallback using the staged helper."""
import ast
from contextlib import contextmanager,nullcontext
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from overlap_order import Queue,Cuda,Buffer

ROOT=Path(__file__).resolve().parent/"decode_overlap"
class ReadyBuffer(Buffer):
    def data_ptr(self):return self.x.data_ptr()
    def stride(self):return self.x.stride()

def main():
    py=(ROOT/"event_dflash_worker_v2.py").read_bytes()
    fn=next(n for n in ast.walk(ast.parse(py)) if isinstance(n,ast.FunctionDef) and n.name=="_project_during_verify")
    count,early=0,0
    for match in (True,False):
        for marker in (True,False):
            for n in (3,4,8):
                for src in ("left","right"):
                    for i in range(6):
                        events=[];side=Queue("projection",events);forward=Queue("target_"+src,events);cuda=Cuda(forward)
                        side.wait_event=lambda e:side.submit("wait_event",wait=e.done)
                        torch_fake=NS(cuda=cuda,bfloat16=torch.bfloat16,inference_mode=nullcontext)
                        scope=dict(torch=torch_fake,contextmanager=contextmanager,draft_tp_context=lambda x:nullcontext())
                        exec(compile(ast.Module([fn],[]),"<early output worker>", "exec"),scope)
                        whole=ReadyBuffer(torch.zeros(8,16,dtype=torch.bfloat16));expected=i+n+1
                        hidden=ReadyBuffer(whole.x[:n] if match else torch.empty(n,16,dtype=torch.bfloat16));old=[]
                        event=NS(done=None);forward.submit("aux_ready",lambda:whole.x.fill_(expected));event.done=forward.tail
                        forward.submit("target_head")
                        if not match:forward.submit("actual post-return copy",lambda:hidden.x.copy_(whole.x[:n]))
                        def transform(h):
                            out=ReadyBuffer(torch.empty(n,3,dtype=torch.bfloat16))
                            side.submit("FC",lambda:out.x.copy_(h.x[:,:3]*2))
                            return out
                        worker=NS(_context_projection_stream=side,draft_parallel_layout=None,
                                  draft_model=NS(fc=NS(in_features=16,weight=NS(device="cuda:0")),project_target_hidden=transform))
                        with scope["_project_during_verify"](worker,hidden,n,True,(event,whole) if marker else None) as out:
                            forward.submit("accept")
                        forward.submit("publish KV",lambda:old.append(out.x.clone()))
                        forward.submit("overwrite",lambda:whole.x.fill_(-123))
                        done=set();trace=[]
                        while len(done)<len(events):
                            ready=[j for j,(_,_,ds,_) in enumerate(events) if j not in done and all(d in done for d in ds)]
                            j=min(ready,key=lambda j:events[j][1]!="projection")
                            op,_,_,cb=events[j];cb();trace.append(op);done.add(j)
                        assert torch.all(old[0]==expected*2)
                        if marker and match:
                            assert trace.index("aux_ready")<trace.index("FC")<trace.index("target_head")
                            early+=1
                        else:
                            assert trace.index("target_head")<trace.index("FC")
                        if not match:assert trace.index("actual post-return copy")<trace.index("FC")
                        count+=1
    output=dict(scope="Adversarial CPU event/DAG simulation of the packaged CUDA event helper. Eligible same-storage prefix slices and post-copy fallbacks on two separate graph event streams.",
                cases=count,early_execution=early,no_unready_reads=True)
    (ROOT/"event_ordering_cpu.json").write_text(json.dumps(output,indent=2));print(output)
if __name__=="__main__":main()
