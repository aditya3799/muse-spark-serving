"""Multi-chunk correctness/timing of scored DFlash prefill with explicit retained-KV checks."""
import __future__,ast,ctypes,hashlib,importlib.util,json,logging,sys,subprocess,time
from array import array
from contextlib import nullcontext
from pathlib import Path
from types import MethodType,SimpleNamespace as S
import torch,os
from entry import stock,MODE,WORK,req
from regions import DFlashRegionCoverage
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
W=2048
CHUNK=16384
END=74160
FORK=32768
SMALL=os.environ.get('OKC_SMALL_WINDOW_REPLAY')=='1'
if SMALL:CHUNK,END,FORK=1024,7953,4096
MIX_AT=3072 if SMALL else 32768
MIX_N=1024 if SMALL else 2048
LABEL='smoke' if SMALL else 'regions'
DELTA=753
EARLY=os.environ.get("OKC_EARLY_TAPS")=="1"
if EARLY:
    assert SMALL, "full producer replay on this GPU is limited to small chunks"
    LABEL="qualified_early"
B=1
BASE="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
ARMS=("baseline","candidate","control")
SCENES=(f"cold_{END}","same_tip_753","inner_fork_753","mixed_eligible_and_short")
SEEDS={"pilot":915137,"holdout":715631}
def read_method(source,owner,name,ns):
    top=ast.parse(source)
    cl=next(x for x in top.body if getattr(x,"name",None)==owner)
    node=next(x for x in cl.body if getattr(x,"name",None)==name)
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),name,"exec",flags=__future__.annotations.compiler_flag),ns)
    return ns[name]
def data(module,path):
    spec=importlib.util.spec_from_file_location(module,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[module]=obj;spec.loader.exec_module(obj)
    return obj
def request(ids):
    r=req(ids);r.full_untruncated_fill_ids=ids
    return r
def setup(seed):
    import faulthandler
    faulthandler.dump_traceback_later(90,repeat=True)
    device=torch.device("cuda")
    gen=torch.Generator(device=device).manual_seed(seed)
    bf=torch.bfloat16
    def rnd(shape,std=1.):
        return torch.empty(shape,dtype=bf,device=device).normal_(std=std,generator=gen)
    print("allocating native FC",seed,flush=True)
    fc=torch.nn.Linear(40960,8192,bias=False,device="meta",dtype=bf)
    fc.weight=torch.nn.Parameter(rnd((8192,40960),40960**-.5),requires_grad=False)
    norm=rnd((8192,))
    lib=ctypes.CDLL(str(ROOT/"research_2026-11-18_lt_audit/rms_linux.so"))
    native=lib.run_bf16_rms; ptr=ctypes.c_void_p
    native.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_float,ptr];native.restype=ctypes.c_int
    def normalize(x):
        out=torch.empty_like(x)
        assert native(x.data_ptr(),norm.data_ptr(),out.data_ptr(),x.shape[0],x.shape[1],1e-5,torch.cuda.current_stream().cuda_stream)==0
        return out
    print("building fused KV",flush=True)
    fused=data("okc_dflash_fused_kv",ROOT/"sglang/python/sglang/kernels/ops/speculative/fused_kv_materialize.py")
    h=object.__new__(fused.FusedKVMaterializeHelper)
    sys.path.insert(0,str(ROOT/"research_2026-11-18_lt_audit"))
    from real_coupled import configured_rope
    vals=dict(device=device,n_layers=5,flat_kv_weight_t=rnd((8192,1280),8192**-.5),layer_out_dim=256,num_kv_heads=1,head_dim=128,rotary_dim=128,
              k_norm_weights=rnd((5,128)),eps_values=torch.full((5,),1e-5,device=device),_workspace_capacity=0,_workspace_dtype=None,
              _proj_workspace=None,_k_workspace=None,_v_workspace=None,_mm_out_supported=True,max_position_hint=131071,
              rotary_emb=S(cos_sin_cache=configured_rope(device,500000)),_reserved_rope_cache_len=131072)
    for name,value in vals.items():setattr(h,name,value)
    h._ensure_workspace(CHUNK,bf)
    model=S(fc=fc,hidden_norm=normalize,quantizable=False,layers=[S(self_attn=S(attn=S(layer_id=i,k_scale=None,v_scale=None))) for i in range(5)])
    model.project_target_hidden=MethodType(read_method(stock("python/sglang/srt/models/dflash.py"),"DFlashDraftModel","project_target_hidden",dict(torch=torch)),model)
    first=stock(WORK)
    second=(HERE/("applied_early" if EARLY else "applied_regions")/WORK).read_text()
    ctx={"torch":torch,"logger":logging.getLogger("okc"),"draft_tp_context":lambda _:nullcontext(),"KVWriteLoc":lambda t,physical:t,
        "CaptureHiddenMode":S(FULL=True),"SpecTpSyncSite":S(DFLASH_TARGET=True),"ForwardMode":S(EXTEND=MODE),"async_h2d":lambda t,**kw:torch.tensor(t,**kw)}
    # Positions are identical CUDA arrays in the fixture's explicit plan; includes mixed prefill.
    ctx["compute_position"]=lambda *args:(active["positions"],None)
    active={"jitter":0}
    basis=torch.empty((1024,40960),device=device,dtype=torch.float32 if EARLY else bf).normal_(generator=gen)
    inp=None if EARLY else torch.empty((max(CHUNK,MIX_N+16),40960),dtype=bf,device=device)
    if EARLY:
        from target_entry import CAP,target as native_target
        ctx["CaptureHiddenMode"]=CAP
    print("allocating worker pools",flush=True)
    idx=torch.arange(END+DELTA+1,device=device)
    perm=torch.randperm(END+DELTA+1,device=device,generator=gen)
    def target(batch,**kwargs):
        index=torch.remainder(active["features"]+active["jitter"],1024)
        torch.index_select(basis,0,index,out=inp[:index.numel()])
        return S(kv_loc_plan=active["locations"],logits_output=S(hidden_states=inp[:index.numel()]),next_token_ids=batch.seq_lens)
    workers={}
    for arm in ARMS:
        pool=torch.zeros((2,5,END+DELTA+1,1,128),dtype=bf,device=device)
        def setbuf(layer,loc,k,v,*_,dst=pool):
            # Identical NHD scatter expressions from pinned _set_kv_buffer_impl fallback.
            dst[0,layer.layer_id,loc]=k;dst[1,layer.layer_id,loc]=v
        worker=S(_validate_phase1_sampling_support=lambda *a:None,_wait_for_fused_hicache_load=lambda *a:None,
            target_worker=S(forward_batch_generation=target),_tp_sync=S(sync=lambda *a:None),model_runner=S(prefill_attention_backend_str="fa3",device=device),
            _draft_write_ids=lambda z:z,_make_next_draft_input_prefill=lambda **kw:kw,draft_model=model,draft_model_runner=S(token_to_kv_pool=S(set_kv_buffer=setbuf)),
            draft_parallel_layout=None,lilicorr=None,_use_fused_kv_materialize=True,_fused_kv_helper=h)
        for meth in ("_append_target_hidden_fused","_append_target_hidden_to_draft_kv_by_loc","forward_batch_generation"):
            setattr(worker,meth,MethodType(read_method(second if arm=="candidate" else first,"DFlashWorkerV2",meth,ctx.copy()),worker))
        if EARLY:
            state=dict(packs=0,gathers=[],monitor=False)
            state["make"]=lambda fb:basis.index_select(0,torch.remainder(active["features"]+active["jitter"],1024)).view(-1,5,8192).permute(1,0,2)
            call=native_target(arm,state)
            def target_source(batch,call=call,state=state,**kwargs):
                state["ids"]=active["positions"];state["loc"]=active["locations"]
                return call(batch,**kwargs)
            worker.target_worker=S(forward_batch_generation=target_source)
            worker._can_omit_target_aux=True
        workers[arm]=worker,pool
    def reform(arm):
        worker,_=workers[arm]
        if arm=="candidate":worker._window_coverage=DFlashRegionCoverage(W,1)
    tokens=array("q",range(END+DELTA+1)); long=request(tokens[:END]);tip=request(tokens[:END+DELTA])
    fork=request(tokens[:FORK]+array("q",[124098])*DELTA); short=request(tokens[:16]);mixlong=request(tokens[:END])
    def run(arm,scene):
        worker,_=workers[arm];cover=getattr(worker,"_window_coverage",None)
        if scene==SCENES[0]: r,p,physical=long,0,0
        elif scene==SCENES[1]:r,p,physical=tip,END,0
        elif scene==SCENES[2]:
            r=fork;p=FORK;physical=0
            if cover:p=min(p,cover.match_limit(r,r.origin_input_ids,p))
        else:r,p,physical=mixlong,MIX_AT,0
        start=p;limit=len(r.origin_input_ids)
        while p<limit:
            n=min(CHUNK,limit-p);pos=idx[p:p+n];features=pos
            if scene==SCENES[2]:features=pos+(pos>=FORK)*101
            locations=perm[p:p+n];reqs=[r];prefs=[p];ext=[n]
            if scene==SCENES[-1] and p==start:
                reqs+=[short];prefs+=[0];ext+=[16]
                # Avoid overlapping run-time writes between the two requests.
                pos=torch.cat((pos,idx[:16]));features=torch.cat((features,idx[:16]+433));locations=torch.cat((locations,perm[-16:]))
            active.update(positions=pos,features=features,locations=locations)
            batch=S(forward_mode=MODE,is_extend_in_batch=False,reqs=reqs,prefix_lens=prefs,extend_lens=ext,out_cache_loc=locations,
                    seq_lens=torch.tensor([i+j for i,j in zip(prefs,ext)],device=device))
            if EARLY:batch.return_hidden_states_mode=CAP.NULL
            batch.hicache_consumer_index=0
            result=worker.forward_batch_generation(batch)
            assert result.logits_output.hidden_states is None
            if scene==SCENES[-1] and p==start:
                active["short_before"]=workers[arm][1][:,:,perm[-16:]].clone()
            if cover:cover.may_publish(r,p+n)
            p+=n
        if scene==SCENES[-1]:
            assert torch.equal(active["short_before"],workers[arm][1][:,:,perm[-16:]])
        return dict(touched=limit-start+(16 if scene==SCENES[-1] else 0),start=start,end=limit)
    def snapshot(arm,scene):
        _,pool=workers[arm]
        end=[END,END+DELTA,FORK+DELTA,END][SCENES.index(scene)]
        ix=perm[end-W:end]
        if scene==SCENES[-1]:ix=torch.cat((ix,perm[-16:]))
        return pool[:,:,ix].contiguous().view(torch.int16)
    def change():
        active["jitter"]=(active["jitter"]+127)%1024
    return reform,run,snapshot,change
def main(stage,rounds):
    seed=SEEDS[stage]
    reform,run,snapshot,change=setup(seed)
    scope="one SM86, original DFlashWorkerV2 prefill, full BF16 projection (40960 to 8192), native FlashInfer RMS, original five-layer fused KV/RoPE, NHD fallback scatter; includes synthetic target-hidden fixture gather, metadata and safety checks. No target transformer, radix implementation, H200, or TP collectives."
    if EARLY:scope+=" Heterogeneous extend runs until both requests can query; checks 16 live short rows and final 2048 live long rows. Real TpModelWorker.forward_batch_generation, Muse target core and outer CausalLM, five FP32 residual fixtures and source packing; no full target transformer."
    detail=dict(seed=seed,orders=[],cases={},complete=False)
    receipt=dict(complete=False,agent="speculation",candidate="early_target_window_region_repair" if EARLY else "window_prefix_region_repair",baseline_sha256=BASE,candidate_sha256=hashlib.sha256((HERE/("early_candidate.diff" if EARLY else "region_candidate.diff")).read_bytes()).hexdigest(),
                 correct=True,active=True,scope=scope,seed=seed,stage=stage,bench=dict(instances=1,shapes={}))
    for scene in SCENES:
        detail["cases"][scene]=dict(mismatches={a:[] for a in ARMS[1:]},compared=[],changed=[],plans={})
        receipt["bench"]["shapes"][scene]=dict(weight=1,times={a:[] for a in ARMS})
    prev={}
    print('setup ready',stage,CHUNK,END,flush=True)
    for step in range(-1,rounds):
        change();torch.cuda.synchronize()
        order=[*ARMS] if step<0 else list(ARMS[step//2%3:]+ARMS[:step//2%3])
        if step>=0 and step%2:order.reverse()
        if step>=0:detail["orders"].append(order)
        for a in ARMS:reform(a)
        for scene in SCENES:
            cold_times={}
            d=detail["cases"][scene]
            for arm in order:
                torch.cuda.synchronize()
                t=time.perf_counter_ns(); meta=run(arm,scene);torch.cuda.synchronize()
                us=(time.perf_counter_ns()-t)/1000
                cold_times[arm]=us
                if step in (-1,0):print(stage,step,scene,arm,round(us/1e6,3),flush=True)
                if step>=0:
                    receipt["bench"]["shapes"][scene]["times"][arm].append(us);d["plans"][arm]=meta
            if step==-1:
                ref=snapshot("baseline",scene)
                warm=detail.setdefault("unscored_warmup",{})
                warm[scene]=dict(synchronized_wall_us=cold_times,compared=ref.numel(),
                    wrong={arm:torch.count_nonzero(snapshot(arm,scene)!=ref).item() for arm in ARMS[1:]})
                (HERE/f"{LABEL}_{stage}_warmup.json").write_text(json.dumps(warm,indent=2))
            if step>=0:
                base=snapshot("baseline",scene);d["compared"].append(base.numel())
                if scene in prev:d["changed"].append(torch.count_nonzero(prev[scene]!=base.reshape(-1)[:8192]).item())
                prev[scene]=base.reshape(-1)[:8192].clone()
                for arm in ARMS[1:]:
                    bad=torch.count_nonzero(snapshot(arm,scene)!=base).item();d["mismatches"][arm].append(bad);receipt["correct"]&=bad==0
        print(stage,step,{k:v["mismatches"]["candidate"][-1:] for k,v in detail["cases"].items()},flush=True)
        if step>=0:
            detail["complete"]=receipt["complete"]=step==rounds-1
            detail["peak_mib"]=torch.cuda.max_memory_allocated()/2**20
            (HERE/f"{LABEL}_{stage}_detail.json").write_text(json.dumps(detail,indent=2))
            (HERE/f"{LABEL}_{stage}.json").write_text(json.dumps(receipt,indent=2))
if __name__=="__main__":
    with torch.inference_mode():
        for stage in (("pilot","holdout") if sys.argv[1]=="both" else (sys.argv[1],)):
            main(stage,int(sys.argv[2]) if len(sys.argv)>2 else 4)
            torch.cuda.empty_cache()

