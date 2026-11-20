"""Measure exact projected context and native rank-local greedy verification fork/join on one device."""
import argparse
import ast
import contextlib
import ctypes
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import MethodType,SimpleNamespace as NS
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/"decode_overlap"
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
BASE="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
SG=ROOT/"sglang/python/sglang"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def method(path,name,scope):
    node=next(n for n in ast.walk(ast.parse(path.read_bytes())) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),scope)
    return scope[name]

def rms():
    dll=ctypes.CDLL(str(ROOT/"research_2026-11-18_lt_audit/rms_linux.so"))
    ptr=ctypes.c_void_p
    call=dll.run_bf16_rms
    call.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_float,ptr]
    call.restype=ctypes.c_int
    def norm(x,weight):
        out=torch.empty_like(x)
        status=call(x.data_ptr(),weight.data_ptr(),out.data_ptr(),*x.shape,1e-5,torch.cuda.current_stream().cuda_stream)
        if status:raise RuntimeError(status)
        return out
    return norm

def main(args):
    import faulthandler
    faulthandler.dump_traceback_later(90,repeat=True)
    print("begin",args.stage,args.batch,flush=True)
    g=torch.Generator(device="cuda").manual_seed(args.seed)
    accept=load("overlap_pinned_accept",SG/"kernels/ops/speculative/dflash.py")._compute_dflash_accept_bonus_triton_unchecked
    select=load("overlap_pinned_select",SG/"kernels/ops/sampling/softcap_row_argmax.py").softcap_row_argmax
    parallel=load("overlap_trial_select",ROOT/"research_2026-11-18_rl/decode_vocab/sharded_max.py").parallel_softcap_max
    scope=dict(torch=torch,contextmanager=contextlib.contextmanager,draft_tp_context=lambda _:contextlib.nullcontext())
    event_mode=args.target_head
    workfile="event_dflash_worker_v2.py" if event_mode else "worker.py"
    overlap=method(OUT/workfile,"_project_during_verify",scope)
    project=method(HERE/"decode_selector/dflash_original_src.py","project_target_hidden",{"torch":torch})
    fc=torch.nn.Linear(40960,8192,bias=False,device="cuda",dtype=torch.bfloat16)
    fc.weight.normal_(std=.01,generator=g)
    head=None
    if event_mode:
        head=torch.nn.Linear(8192,25256,bias=False,device="cuda",dtype=torch.bfloat16)
        head.weight.normal_(std=.01,generator=g)
        geom=ROOT/"release_2026-11-18/candidate_sources/python/sglang/srt/models/muse_spark.py"
        target_head=method(geom,"_compute_lm_head",dict(torch=torch,use_intel_amx_backend=lambda _:False))
        target=NS(use_fp32_output=True)
    scale=torch.rand(8192,device="cuda",generator=g).bfloat16();kernel=rms()
    model=NS(fc=fc,hidden_norm=lambda x:kernel(x,scale),quantizable=False)
    model.project_target_hidden=MethodType(project,model)
    worker=NS(draft_model=model,_context_projection_stream=torch.cuda.Stream(),draft_parallel_layout=None)
    forward=torch.cuda.Stream()
    forward.wait_stream(torch.cuda.current_stream())
    doc=dict(agent="speculation",candidate="overlapped_project_and_verify_"+args.selection,baseline_sha256=BASE,
             candidate_sha256=hashlib.sha256((OUT/"candidate.diff").read_bytes()).hexdigest(),
             seed=args.seed,scope="One local SM86. Native dense 40960→8192 BF16 FC and native RMS, local vocabulary select, 8-way post-gather winner and genuine pinned accept kernel. Seven remote rank maxima supplied; network, target/draft forward and KV append outside timing. Native operands are synthetic.",
             active=True,correct=True,complete=False,bench=dict(instances=1,shapes={}))
    if event_mode:
        doc["candidate"]="captured_aux_event_"+args.selection
        doc["candidate_sha256"]=hashlib.sha256((OUT/"event_candidate.diff").read_bytes()).hexdigest()
        doc["scope"]="One SM86, actual parent target BF16 8192?25256 shard with FP32 output and scaling; real target graph and external aux-ready event, actual native dense 40960?8192 FC/RMS, pinned deferred local greedy selection and accept. Seven remote maxima are supplied. Synthetic weights, no eight-rank communication or KV publication time."
    print("weights ready",flush=True)
    detail={}
    path=OUT/f"{args.stage}_{args.selection}{'_event' if event_mode else ''}.json"
    with torch.cuda.stream(forward):
        for b in args.batch:
            print("shape begin",b,flush=True)
            m=8*b
            h=torch.empty((m,40960),dtype=torch.bfloat16,device="cuda")
            logits=torch.empty((m,25256),device="cuda")
            graphs={}
            if event_mode:
                packed_input=torch.empty_like(h)
                head_input=torch.empty((m,8192),dtype=torch.bfloat16,device="cuda")
                head_input.normal_(generator=g);packed_input.normal_(generator=g)
                def head_forward(mark):
                    h.copy_(packed_input)
                    if mark is not None:mark.record()
                    z=target_head(target,head_input,head)
                    if args.selection=="native":z.mul_(.1767766952966369)
                    return z
                head_forward(None);forward.synchronize()
                for tag in ("baseline","candidate"):
                    e=torch.cuda.Event(external=True) if tag=="candidate" else None
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): z=head_forward(e)
                    graphs[tag]=(graph,z,e)
                    print("captured",b,tag,flush=True)
            d=torch.empty((b,8),dtype=torch.int64,device="cuda")
            allrank=torch.empty((8,m,2),device="cuda")
            allrank[:,:,0].fill_(-float("inf"));allrank[:,:,1].fill_(0)
            # Force a known highest token on alternating rows. All seven remote rank
            # scores are already available and remain subordinate in this fixture.
            ranks=torch.arange(m,device="cuda")
            prefix=torch.arange(b,dtype=torch.int32,device="cuda")+1300
            def fill(i):
                if event_mode:
                    packed_input.normal_(std=.3,generator=g);head_input.normal_(std=.4,generator=g)
                    graph,z,_=graphs["baseline"];graph.replay()
                    # Reproduce the current target's global winner in the seven
                    # local proposal slots to test alternating full/partial acceptance.
                    base=z if args.selection=="native" else z*.1767766952966369
                    chosen=select(base,20.,25256,0)[:,1].long()
                else:
                    h.normal_(std=.3,generator=g);logits.normal_(generator=g)
                    chosen=(ranks*137+4+i*19)%25256
                    logits[ranks,chosen]=120.
                d[:,0]=1;d[:,1:]=chosen.view(b,8)[:,:7]
                if i%2:d[::2,2]=25257
            def call(side):
                values=logits
                if event_mode:
                    graph,values,e=graphs["candidate" if side else "baseline"];graph.replay()
                    guard=overlap(worker,h,m,active=side,ready=(e,h) if side else None)
                else:
                    guard=overlap(worker,h,m,active=side)
                with guard as z:
                    factor=.1767766952966369 if event_mode else 1.
                    local=(select(values,20.,25256,0) if args.selection=="native" else parallel(values,20.,25256,0,scale=factor))
                    allrank[0].copy_(local)
                    best=allrank[...,0].argmax(0,keepdim=True)
                    target=allrank[...,1].gather(0,best).squeeze(0).long().view(b,8)
                    a=torch.empty_like(prefix);c=torch.empty_like(prefix);bonus=torch.empty(b,device="cuda",dtype=torch.long)
                    out=torch.empty_like(d);new=torch.empty_like(prefix)
                    accept(d,target,a,c,bonus,out,prefix,new)
                if z is None:z=model.project_target_hidden(h)
                return (z,target,a,c,bonus,out,new,values) if event_mode else (z,target,a,c,bonus,out,new)
            fill(0);call(False);call(True);torch.cuda.synchronize()
            print("warmup complete",b,flush=True)
            times={k:[] for k in ("baseline","candidate","control")};changed=0;prev=None;full=0;partial=0
            wrong={arm:{} for arm in ("candidate","control")}
            for i in range(args.rounds):
                if i==0:print("paired timing",b,flush=True)
                fill(i);torch.cuda.synchronize(); results={}
                seq=("baseline","candidate","control") if i%2==0 else ("control","candidate","baseline")
                for arm in seq:
                    t=time.perf_counter();z=call(arm=="candidate");forward.synchronize()
                    times[arm].append((time.perf_counter()-t)*1e6)
                    results[arm]=tuple(y.clone() for y in z)
                ref=results["baseline"]
                for arm in ("candidate","control"):
                    for label,x,y in zip(("ctx","target_argmax","accept","commit","bonus","output_ids","new_lens","target_logits"),ref,results[arm]):
                        if not torch.equal(x,y):
                            doc["correct"]=False
                            wrong[arm][label]=wrong[arm].get(label,0)+int((x!=y).sum())
                if prev is not None:changed+=int(not torch.equal(prev,ref[0]))
                prev=ref[0]
                full+=int((ref[3]==8).sum());partial+=int((ref[3]<8).sum())
            doc["correct"]&=changed==args.rounds-1 and full>0 and partial>0
            doc["bench"]["shapes"][f"B{b}"]=dict(weight=1,times=times)
            detail[f"B{b}"]=dict(changed_input_output_pairs=changed,full_commits=full,partial_commits=partial,different_elements=wrong)
            doc["detail"]=detail
            path.write_text(json.dumps(doc,indent=2));print("completed",b,doc["correct"],flush=True)
    doc["complete"]=True;path.write_text(json.dumps(doc,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True);p.add_argument("--batch",type=int,nargs="+",default=[1,4,16,32]);p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--stage",choices=["pilot","holdout"],required=True);p.add_argument("--selection",choices=["native","parallel"],default="native")
    p.add_argument("--target-head",action="store_true")
    with torch.inference_mode():main(p.parse_args())
