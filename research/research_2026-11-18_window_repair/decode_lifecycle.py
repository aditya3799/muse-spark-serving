"""Check capture-hook staging, original graph-return selection, live references, and checkpoint confidence."""
from __future__ import annotations
import ast,__future__,copy,faulthandler,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from types import MethodType,SimpleNamespace as NS
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;D=H/"decode_selector";sys.path.insert(0,str(D))
import direct_walk,original_walk
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
class Output:
    def __init__(self,h):self.hidden_states=h
class InterpreterDevice(torch.Tensor):
    # In the interpreter-only run, dispatch to the CUDA operator on CPU storage.
    @property
    def is_cuda(self):return True
def node(src,name,ns,cl=None):
    t=ast.parse(src)
    if cl:t=next(n for n in t.body if getattr(n,"name",None)==cl)
    n=next(n for n in t.body if getattr(n,"name",None)==name);n.decorator_list=[]
    exec(compile(ast.Module(body=[n],type_ignores=[]),name,"exec",flags=__future__.annotations.compiler_flag),ns)
    return ns[name]
def native(path):return subprocess.check_output(["git","-C",str(H.parent/"sglang"),"show",PIN+":"+path]).decode()
def graph_return(src):
    tree=ast.parse(src)
    cl=next(n for n in tree.body if getattr(n,"name",None)=="DFlashWorkerV2")
    fn=next(n for n in cl.body if getattr(n,"name",None)=="forward_batch_generation")
    branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=="self._draft_sampler is not None and draft_out.can_run_graph")
    block=copy.deepcopy(branch);block.orelse=[]
    wrap=ast.parse("def graph_return(self,batch,draft_out,bs):\n    self._selector_sample = None\n    return draft_next").body[0]
    wrap.body.insert(1,block);m=ast.fix_missing_locations(ast.Module(body=[wrap],type_ignores=[]))
    scope=dict(_is_all_greedy=lambda s:s is None or s.is_all_greedy)
    exec(compile(m,"pinned_worker_graph_return","exec"),scope);return scope["graph_return"]
def weights(device):
    d=json.loads((D/"real_selector_weights.json").read_text());p=Path(d["file"])
    if os.name!="nt":
        p=Path(subprocess.check_output(["wslpath","-u",d["file"]],text=True).strip())
        if not p.exists():p=next(Path("/mnt/c/Users").glob("*/AppData/Local/Temp/"+p.name))
    blob=p.read_bytes();r={}
    for key,x in d["slices"].items():
        raw=blob[x["offset"]:x["offset"]+x["length"]]
        assert hashlib.sha256(raw).hexdigest()==x["sha256"]
        r[key.split("candidate_selector.")[-1]]=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(x["shape"]).to(device)
    feat=[k for k in d["all_tensor_names"] if any(q in k.lower() for q in ("confidence","fore_d","forehead","acceptance"))]
    return r,feat
def main(device):
    interp=device=="cpu";torch.set_num_threads(2)
    start_time=time.monotonic()
    def progress(stage):
        message=dict(stage=stage,elapsed_s=round(time.monotonic()-start_time,3))
        (D/f"lifecycle_{device}_progress.json").write_text(json.dumps(message))
        print(stage,message["elapsed_s"],flush=True)
    if not interp: faulthandler.dump_traceback_later(45,repeat=True)
    progress("start")
    g=torch.Generator(device=device).manual_seed(24581)
    w,heads=weights(device);A,B=w["predecessor_codebook"],w["successor_codebook"];projection=w["hidden_projection.weight"]
    progress("real_weights_ready")
    b,k,h=4,16,8192
    hs=torch.randn((b*8,h),dtype=torch.bfloat16,device=device,generator=g)
    anchor=torch.randint(4096,(b*8,),device=device,generator=g)
    ids=torch.randn((b,7,4096),device=device,generator=g).topk(16,dim=-1).indices
    unary=torch.randn((b,7,16),device=device,generator=g)
    progress("lattice_inputs_ready")
    hook_src=native("python/sglang/srt/speculative/draft_worker_common.py")
    make_hook=node(hook_src,"make_draft_sampler_capture_hook",{"LogitsProcessorOutput":Output})
    info=lambda:None
    state={};buffers={};saved={};rows=[];samples=[];logs={}
    for arm in ("baseline","candidate"):
        impl=original_walk if arm=="baseline" else direct_walk
        tag="original" if arm=="baseline" else "direct"
        wm=(D/f"dflash_worker_v2_{tag}_src.py").read_text(encoding="utf8")
        dm=(D/f"dflash_{tag}_src.py").read_text(encoding="utf8")
        score=node(dm,"_score_edges",dict(torch=torch))
        source=dict(torch=torch,F=F,_score_edges=score,selector_walk_triton=impl.selector_walk_triton)
        sel=NS(top_k=k,predecessor_codebook=A,successor_codebook=B,hidden_projection=lambda x:F.linear(x,projection))
        sel.build_lattice=MethodType(node(dm,"build_lattice",source.copy(),"CandidateSelector"),sel)
        sel.sample_path=MethodType(node(dm,"sample_path",source.copy(),"CandidateSelector"),sel)
        if arm=="candidate":sel.sample_path_into=MethodType(node(dm,"sample_path_into",source.copy(),"CandidateSelector"),sel)
        dflash=NS(candidate_selector=sel,compute_candidates=lambda h:(ids.view(-1,k),unary.view(-1,k)))
        lattice=node(wm,"_selector_lattice",{})
        def run_lattice(*args,arm=arm,lattice=lattice):
            idx,v=lattice(*args)
            logs[arm]=v
            return idx,(v.as_subclass(InterpreterDevice) if interp else v)
        cls=node(wm,"_SelectorDraftSampler",dict(torch=torch,_selector_lattice=run_lattice,resolve_greedy_mask=lambda *,bs,sampling_info,device:(sampling_info.top_ks<=1).view(-1)))
        sampler=cls(draft_model=dflash,block_size=8,max_bs=b,device=device,sampling_enabled=True)
        hook=make_hook(sampler)
        worker=NS(block_size=8,_draft_sampler=sampler,selector=sel,lilicorr=None,_selector_sampling_enabled=True)
        forward_batch=NS(input_ids=anchor)
        def invoke(hook=hook,fb=forward_batch):
            out=Output(hs);hook(None,out,fb,hs.shape[0]);return out
        if not interp:
            progress(arm+"_before_warmup")
            warm_stream=torch.cuda.Stream()
            warm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm_stream):invoke()
            torch.cuda.synchronize()
            progress(arm+"_warmup_done")
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=invoke()
            invoke=graph.replay
            progress(arm+"_captured")
        state[arm]=(sampler,worker,invoke,graph_return(wm))
    uniform_prev={};prev_token=None
    for step in range(12):
        ids.add_(7).remainder_(4096);unary.copy_(unary.roll(3,-1));hs.mul_(-1)
        mode=("greedy","mixed","sample","none")[step%4]
        allgreedy=mode in ("greedy","none")
        ks=torch.ones(b,device=device,dtype=torch.int32)
        if mode=="mixed":ks[::2]=11
        if mode=="sample":ks.fill_(11)
        info=None if mode=="none" else NS(top_ks=ks,temperatures=torch.linspace(.2,1.7,b,device=device),is_all_greedy=allgreedy)
        results={}
        for arm,(sampler,worker,invoke,ret) in state.items():
            sampler.stage_sampling_params(bs=b,sampling_info=info)
            # Rewind per arm to compare equal RNG histories, then replay twice
            # without rewinding. The second replay must draw new uniforms.
            if interp:torch.manual_seed(88213+step)
            else:torch.cuda.manual_seed(88213+step)
            invoke();u1=sampler.uniforms.clone();invoke();u2=sampler.uniforms.clone()
            assert not torch.equal(u1,u2)
            tokens=ret(worker,NS(sampling_info=info),NS(can_run_graph=True),b)
            assert tokens.data_ptr()==sampler.out.data_ptr()
            assert (worker._selector_sample is None)==allgreedy
            if not allgreedy:
                px,qx=worker._selector_sample
                assert px.data_ptr()==sampler.candidate_out.data_ptr() and qx.data_ptr()==sampler.q_out.data_ptr()
                assert torch.equal(px,ids)
            addresses=tuple(t.data_ptr() for t in (sampler.out,sampler.candidate_out,sampler.q_out))
            if arm in buffers:
                assert addresses==buffers[arm] and torch.equal(saved[arm],tokens)
            saved[arm]=tokens;buffers[arm]=addresses
            results[arm]=(tokens.clone(),sampler.candidate_out.clone(),sampler.q_out.clone(),u1,u2)
        for x,y in zip(results["baseline"],results["candidate"]):
            assert torch.equal(x,y)
        tokens,current_ids,actual_q,_,_=results["candidate"]
        assert prev_token is None or not torch.equal(prev_token,tokens)
        prev_token=tokens.clone()
        # Rank a realized proposal using the untempered original selector's
        # actual edge logits. Greedy q itself is a point mass by definition.
        v=logs["baseline"];ix=(current_ids==tokens[:,:,None]).to(torch.int32).argmax(-1);p=torch.zeros(b,dtype=torch.int64,device=device)
        # Only transfer selected observation vectors after the loop.
        predictions=[];chosen_q=[]
        for j in range(7):
            edge=v[torch.arange(b,device=device),j,p,:].float()
            prob=edge.softmax(-1);p=ix[:,j]
            predictions.append(prob.gather(-1,p[:,None]).reshape(-1))
            chosen_q.append(actual_q[:,j].gather(-1,p[:,None]).reshape(-1))
        pred=torch.cat(predictions).tolist();q_chosen=torch.cat(chosen_q).tolist()
        samples.append(dict(mode=mode,selected_raw_softmax=pred,selected_draft_q=q_chosen))
        progress("verified_step_"+str(step+1))
    result=dict(scope="Pinned host staging, actual capture hook, actual CandidateSelector methods and upper worker graph-hit AST; real selector projection and 4096 BF16 codebook rows, synthetic hidden activations/unary top-16s",
        mode=("Triton CPU interpreter and simulated graph replay" if interp else "actual torch CUDA graph capture/replay"),
        steps=len(samples),changed_every_step=True,distinct_rng_successive_calls=True,static_addresses=True,all_tokens_q_ids_equal=True,
        training_confidence_heads_in_checkpoint=heads,fixtures=samples)
    (D/("lifecycle_cpu.json" if interp else "lifecycle_cuda.json")).write_text(json.dumps(result,indent=2))
    print("passed",device,len(samples),heads,{m:[min(s["selected_raw_softmax"]),max(s["selected_raw_softmax"])] for m,s in ((q["mode"],q) for q in samples[:4])})
if __name__=="__main__":
    with torch.inference_mode():main(sys.argv[1])


