"""Future H200-only native FA3 component gate; default is CPU source preflight.

Runs exact source-extracted verify/decode call expressions, scheduler method, and
FullCudaGraphBackend capture/replay methods. This is NOT a full ModelRunner or
multi-rank serving validation. Installed binary precision must be independently
audited and hash-bound before --run is admitted. Never runs on SM86.
"""
import argparse,contextlib,hashlib,json,math,time,types
from pathlib import Path
import fa3_split_policy as pol
from source_extract import ROOT,FA,BASE,GRAPH,method,make_forward,attention_call
P=Path(__file__).resolve().parent
WRAPPER=ROOT/'sglang/python/sglang/kernels/aot/python/sgl_kernel/flash_attn.py'
REVISION='f89bc2306632d1ec5f97b014dded4254f5b4a907'
N=types.SimpleNamespace

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_contract(c):
    wanted={'source_revision':REVISION,'qkv_dtype':'FP8_E4M3','probability_dtype':'FP8','score_accum_dtype':'FP32','output_partial_dtype':'FP32','lse_partial_dtype':'FP32','combine_dtype':'FP32','all_valid_keys':True,'exp2_unchanged':True,'audited':True}
    for k,v in wanted.items():
        if c.get(k)!=v:raise RuntimeError('native precision contract missing/mismatched '+k)
    for k in ('binary_sha256','wrapper_sha256'):
        if not isinstance(c.get(k),str) or len(c[k])!=64 or any(x not in '0123456789abcdef' for x in c[k]):raise RuntimeError('explicit audited '+k+' required')
    if not c.get('audit_evidence'):raise RuntimeError('native precision audit evidence required')
def preflight():
    m=json.loads((P/'OVERLAY_MANIFEST.json').read_text())
    for rel,digest in m['sources'].items():
        # Builder hashes normalized UTF-8 source text, not platform newline bytes.
        assert hashlib.sha256((ROOT/'sglang'/rel).read_text().encode()).hexdigest()==digest,rel
    calls={phase:attention_call(phase)[1] for phase in ('verify','decode')}
    assert 'num_splits=self.num_splits' in calls['verify']
    assert 'self.decode_num_splits' in calls['decode'] and 'scheduler_metadata=sched_meta' in calls['decode']
    cmake=(ROOT/'sglang/python/sglang/kernels/aot/CMakeLists.txt').read_text();assert REVISION in cmake
    return {'status':'CPU_PREFLIGHT_ONLY_NATIVE_UNRUN','source_revision':REVISION,'source_sha256':{str(x.relative_to(ROOT)):sha(x) for x in (FA,BASE,GRAPH,WRAPPER)},'call_expressions':calls,'splits':[0,1,2,4,8],
      'logical_partial_bytes_max_fixture':4*8*16*8*16*(64+1),
      'holds':['audited installed binary SHA/precision','actual H200 execution','full backend constructor and ModelRunner integration','all-rank immutable policy setup and graph invalidation','serving losslessness and speed'],
      'scope':'Native gate uses real installed FA3, CUDA graphs, exact source call expressions/scheduler and full graph methods; metadata fixture and runner services are per-rank component adapters.'}
class Mode:
    def __init__(self,p):self.p=p
    def is_target_verify(self):return self.p=='verify'
    def is_decode_or_idle(self):return self.p=='decode'
    def is_draft_extend_v2(self):return False

def native(args,report):
    contract=json.loads(Path(args.contract).read_text(encoding='utf-8-sig'));validate_contract(contract)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('H200 CUDA required')
    dev=torch.cuda.current_device()
    if torch.cuda.get_device_capability(dev)!=(9,0) or 'H200' not in torch.cuda.get_device_name(dev).upper():raise RuntimeError('H200 only; SM86 cannot qualify this path')
    import sgl_kernel.flash_attn as native_fa
    if sha(native_fa.__file__)!=contract['wrapper_sha256'] or sha(native_fa.flash_ops.__file__)!=contract['binary_sha256']:raise RuntimeError('installed native wrapper/binary audit mismatch')
    if contract['wrapper_sha256']!=sha(WRAPPER):raise RuntimeError('wrapper does not match the pinned integration source')
    if torch.cuda.mem_get_info(dev)[0]<768*1024**2:raise RuntimeError('requires fixture headroom plus 512MiB free reserve')
    torch.manual_seed(170518);baseline_allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    compute,compute_pin=method(FA,'FlashAttentionBackend','_compute_scheduler_metadata',{})
    ext,ext_pin=make_forward('verify',native_fa.flash_attn_with_kvcache)
    dec,dec_pin=make_forward('decode',native_fa.flash_attn_with_kvcache)
    class Backend:
        _compute_scheduler_metadata=pol.fa3_scheduler_guard(compute)
        forward_extend=pol.fa3_split_scope('forward')(ext)
        forward_decode=pol.fa3_split_scope('forward')(dec)
        @pol.fa3_split_scope('metadata')
        def refresh(self,fb):
            md=self.forward_metadata
            if fb.forward_mode.is_decode_or_idle():
                sched=self._compute_scheduler_metadata(self.B,self.L,md.cache_seqlens_int32,md.cu_seqlens_q)
                n=sched.numel();self._sched_meta_buf[:n].copy_(sched);self._sched_meta_buf[n:].zero_()
                pol.record_scheduler_copy(self,sched,self._sched_meta_buf,n)
                if md.scheduler_metadata is None:md.scheduler_metadata=self._sched_meta_buf[:n]
    null=contextlib.nullcontext
    gns={'torch':torch,'graph_pool_capture_scope':null,'graph_pool_replay_scope':null}
    gm={name:method(GRAPH,'FullCudaGraphBackend',name,gns)[0] for name in ('capture_one','replay','can_run','cleanup')}
    FullCudaGraphBackend=type('FullCudaGraphBackend',(),{name:deco(gm[name]) for name,deco in [('capture_one',pol.fa3_graph_capture_guard),('replay',pol.fa3_graph_replay_guard),('can_run',pol.fa3_graph_can_run_guard),('cleanup',pol.fa3_graph_cleanup_guard)]})
    report.update(status='NATIVE_RUNNING',device=torch.cuda.get_device_name(dev),binary_sha256=contract['binary_sha256'],wrapper_sha256=contract['wrapper_sha256'],checks=[],timings=[],source_fragments={'compute':compute_pin,'verify':ext_pin,'decode':dec_pin})
    outdir=P/args.output
    if not outdir.resolve().is_relative_to(P):raise RuntimeError('output must remain in research lane')
    outdir.mkdir(exist_ok=False,parents=True)
    def checkpoint():
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();(outdir/'results.json').write_text(json.dumps(report,indent=2))
    def create(phase,S,B,L,window=-1):
        Q=8 if phase=='verify' else 1;pages=math.ceil(L/256);blocks=B*pages
        b=Backend();b.num_splits=b.decode_num_splits=0;b.fa_impl_ver=3;b.is_draft_runner=False;b.use_mla=False;b._disable_scheduler_metadata_precompute=False
        b.head_dim=64;b.num_attention_heads=16;b.num_kv_heads=2;b.kv_cache_dtype=torch.float8_e4m3fn;b.page_size=256;b.has_softcap=False;b.B=B;b.L=L;b._get_scheduler_metadata=native_fa.get_scheduler_metadata
        q=torch.zeros((B*Q,16,64),device='cuda',dtype=torch.float8_e4m3fn)
        b.key_cache=torch.zeros((blocks,256,2,64),device='cuda',dtype=torch.float8_e4m3fn);b.value_cache=torch.zeros_like(b.key_cache)
        page=torch.arange(blocks,device='cuda',dtype=torch.int32).view(B,pages)
        b.forward_metadata=N(page_table=page,cache_seqlens_int32=torch.full((B,),L,device='cuda',dtype=torch.int32),cu_seqlens_q=torch.arange(B+1,device='cuda',dtype=torch.int32)*Q,cu_seqlens_k=None,max_seq_len_q=Q,scheduler_metadata=None)
        b._sched_meta_buf=torch.zeros((1+math.ceil(B/4)*4*4,),device='cuda',dtype=torch.int32)
        b.native_kwargs={x:torch.ones((B,2),device='cuda',dtype=torch.float32) for x in ('q_descale','k_descale','v_descale')}
        policy=pol.TargetFA3SplitPolicy(verify=S,decode=S,enabled=True,collective_setup_agreed=True,precision_binary_audited=True)
        owner=N(_research_fa3_split_policy=policy,is_draft_worker=False);b._research_fa3_controller=pol.SplitController(b,owner,policy)
        fb=N(forward_mode=Mode(phase),batch_size=B);layer=N(tp_q_head_num=16,head_dim=64,v_head_dim=64,scaling=.125,logit_cap=0.,window_size=(-1,-1) if window<0 else (window,0),is_swa=window>=0)
        runner=N(model_runner=owner,_replay_attn_backend=lambda:b,enable_pdmux=False,enable_two_batch_overlap=False,ragged_verify_mode=False,_metadata_glue=None,enable_profile_cuda_graph=False)
        g=FullCudaGraphBackend();runner.backend=g;g._cuda_graph_runner=runner;g._graphs={};g._outputs={};g._device_module=torch.cuda;g._tp_group=N(barrier=lambda:None);g._precarve=N(measure=null,mint=lambda:None);g._reuse_output_buffer=False;g._memory_saver_adapter=None;g._pool=torch.cuda.graph_pool_handle();g._capture_stream=None;g._output_buffer=None
        forward=b.forward_extend if phase=='verify' else b.forward_decode
        def call():return forward(q,None,None,layer,fb)
        if torch.cuda.mem_get_info(dev)[0]<512*1024**2:raise RuntimeError('512MiB reserve lost before capture')
        if torch.cuda.memory_allocated()-baseline_allocated>2_000_000_000:raise RuntimeError('conservative decimal 2GB fixture allocation cap exceeded; full-device overhead remains unqualified')
        b.refresh(fb);g.capture_one('fixture',call)
        def replay():
            b.refresh(fb);g._outputs['fixture'].fill_(float('nan'));r=g.replay('fixture',fb)
            return r
        return N(b=b,g=g,q=q,fb=fb,call=call,replay=replay,Q=Q,L=L,B=B,window=window)
    def errors(value,reference):
        delta=value.double()-reference.double()
        return {'max_abs':delta.abs().max().item(),'rms':delta.square().mean().sqrt().item(),'max_reference_abs':reference.double().abs().max().item(),'different_elements':torch.count_nonzero(delta).item()}
    def fp64_reference_first_request(f):
        # Ideal attention on the exact quantized FP8 input values, with all descales1.
        # FP64 softmax/P is an accuracy oracle, not a proposed serving precision path.
        length=int(f.b.forward_metadata.cache_seqlens_int32[0].item())
        pages=f.b.forward_metadata.page_table[0].long()
        key=f.b.key_cache[pages].reshape(-1,2,64)[:length].double()*f.b.native_kwargs['k_descale'][0].double().view(1,2,1)
        value=f.b.value_cache[pages].reshape(-1,2,64)[:length].double()*f.b.native_kwargs['v_descale'][0].double().view(1,2,1)
        query=f.q[:f.Q].double().reshape(f.Q,2,8,64)
        scores=torch.einsum('qghd,kgd->ghqk',query,key)*.125
        valid=torch.arange(length,device='cuda')[None,:] <= (torch.arange(f.Q,device='cuda')+length-f.Q)[:,None]
        if f.window>=0:
            valid &= torch.arange(length,device='cuda')[None,:] >= (torch.arange(f.Q,device='cuda')+length-f.Q-f.window)[:,None]
        scores.masked_fill_(~valid[None,None,:,:],float('-inf'))
        probability=torch.softmax(scores,dim=-1)
        return torch.einsum('ghqk,kgd->qghd',probability,value).reshape(f.Q,16,64)
    fixtures=[]
    try:
        # Exhaustively retain each of 257 legal keys across a tile/page boundary.
        # Uniform scores and one-hot V turn coverage/causality into an exact sign/zero test.
        for phase in ('verify','decode'):
            for window,length,keys in ((-1,257,range(257)),(2047,2049,(0,1,2,7,8,2040,2041,2047,2048))):
                for S in (0,1,2,4,8):
                    f=create(phase,S,1,length,window);fixtures.append(f)
                    for key in keys:
                        f.b.value_cache.zero_();f.b.value_cache[key//256,key%256,:,0]=1
                        r=f.replay().view(f.Q,16,64);torch.cuda.synchronize()
                        position=torch.arange(f.Q,device='cuda')+length-f.Q
                        expected=position>=key
                        if window>=0:expected &= position-window<=key
                        assert torch.isfinite(r).all() and torch.equal(r[:,:,0]>0,expected[:,None].expand(-1,16))
                        assert torch.count_nonzero(r[:,:,1:])==0
                    report['checks'].append({'phase':phase,'requested_splits':S,'window':window,'L':length,'tested_keys':len(keys),'exhaustive':len(keys)==length,'finite_and_causal_window_coverage':True})
                    f.g.cleanup();fixtures.remove(f);del f
                    checkpoint()
        # Captured pointer stability under changed Q/K/V, page permutation and lengths.
        for phase in ('verify','decode'):
            for B,L,window in ((1,2049,-1),(4,4096,-1),(16,8192,-1),(1,2049,2047),(4,4096,2047)):
                row=[];group=[create(phase,S,B,L,window) for S in (0,1,2,4,8)];fixtures+=group
                for generation in range(3):
                    ref=group[0];q=(torch.randn(ref.q.shape,device='cuda')*.2).to(torch.float8_e4m3fn);k=(torch.randn(ref.b.key_cache.shape,device='cuda')*.2).to(torch.float8_e4m3fn);v=(torch.randn(ref.b.value_cache.shape,device='cuda')*.2).to(torch.float8_e4m3fn)
                    lens=torch.tensor([L-i*(generation+1)%min(256,L-8) for i in range(B)],device='cuda',dtype=torch.int32)
                    pages=ref.b.forward_metadata.page_table;perm=torch.randperm(pages.numel(),device='cuda',dtype=torch.int64).to(torch.int32).view_as(pages)
                    results=[];replay_diagnostics=[]
                    for f in group:
                        f.b.native_kwargs['k_descale'].fill_(2.**(generation-1));f.b.native_kwargs['v_descale'].fill_(2.**(1-generation))
                        f.q.copy_(q);f.b.key_cache.copy_(k);f.b.value_cache.copy_(v);f.b.forward_metadata.cache_seqlens_int32.copy_(lens);f.b.forward_metadata.page_table.copy_(perm)
                        r=f.replay();torch.cuda.synchronize();assert torch.isfinite(r).all();results.append(r.clone())
                        eager=f.call();torch.cuda.synchronize();assert torch.isfinite(eager).all()
                        replay_diagnostics.append({'bitwise':torch.equal(r,eager),'error':errors(r,eager),'scope':'same-policy diagnostic; stock nondeterminism is not a precision violation'})
                    fp64=fp64_reference_first_request(group[0])
                    for index,(S,r) in enumerate(zip((0,1,2,4,8),results)):
                        row.append({'generation':generation,'requested_splits':S,'versus_heuristic':errors(r,results[0]),'bitwise_vs_heuristic':torch.equal(r,results[0]),'versus_FP64_first_request':errors(r.view(B,ref.Q,16,64)[0],fp64),'same_policy_replay_diagnostic':replay_diagnostics[index]})
                report['checks'].append({'phase':phase,'B':B,'L':L,'window':window,'changed_inputs_lengths_pages':row,'error_tolerance':'recorded only; no invented serving tolerance'})
                if args.timing:
                    # Component graph timing includes scheduler refresh and output poison only outside measured events.
                    for f in group:
                        deadline=time.monotonic()+.3
                        while time.monotonic()<deadline:f.b.refresh(f.fb);f.g.replay('fixture',f.fb)
                    samples=[[] for _ in group]
                    for order in ([0,1,2,3,4,4,3,2,1,0],)*5:
                        for idx in order:
                            f=group[idx];f.b.refresh(f.fb);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                            for _ in range(10):f.g.replay('fixture',f.fb)
                            end.record();end.synchronize();samples[idx].append(start.elapsed_time(end)*100)
                    report['timings'].append({'phase':phase,'B':B,'L':L,'window':window,'samples_us_by_requested_splits':dict(zip(map(str,(0,1,2,4,8)),samples)),'scope':'attention graph only; scheduler refresh excluded; symmetric order, inspect drift before interpreting'})
                for f in group:f.g.cleanup();fixtures.remove(f)
                checkpoint()
        report['status']='NATIVE_COVERAGE_FINITE_LIFECYCLE_PASS_NUMERICAL_QUALITY_AND_SERVING_UNQUALIFIED';report['numerical_quality_status']='FP64 and replay error metrics reported; no organizer serving threshold invented';checkpoint()
    except Exception as exc:
        report['status']='FAILED';report['error']=repr(exc);checkpoint();raise
    finally:
        for f in fixtures:f.g.cleanup()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--contract',default=str(P/'NATIVE_PRECISION_CONTRACT.template.json'));ap.add_argument('--output',default='native_01');ap.add_argument('--timing',action='store_true');args=ap.parse_args()
    report=preflight()
    if args.run:native(args,report)
    else:(P/'NATIVE_PREFLIGHT.json').write_text(json.dumps(report,indent=2));print(report['status'])
if __name__=='__main__':main()
