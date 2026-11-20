"""Bounded native FA3 SWA scheduler diagnostic; no model integration or compilation."""
import argparse, hashlib, importlib.util, inspect, json, statistics, time, traceback
from pathlib import Path
from types import SimpleNamespace as NS
P=Path(__file__).resolve().parent
HELPER=P/'candidate_sources/python/sglang/srt/layers/attention/fa3_swa_scheduler.py'
HELPER_SHA='167a9e774f549c84c215bba8e00acd9b46a2c00c41d0ee600323fb9163200e11'
WRAPPER_SHA='610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main(args):
    report=dict(status='running',passed=False,rows=[],scope='Native installed FA3 BF16 diagnostic only: exact isolated helper, no model graph, no FP8/H200 claim, no fallback backend or compilation.',started=time.time())
    args.out.mkdir(parents=True,exist_ok=True)
    def save():
        temp=args.out/'results.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.out/'results.json')
    save()
    try:
        import torch
        if not torch.cuda.is_available():
            report.update(status='unsupported',reason='CUDA unavailable');save();return 3
        device=torch.cuda.current_device();props=torch.cuda.get_device_properties(device)
        report['device']=dict(name=props.name,capability=list(torch.cuda.get_device_capability(device)),torch=torch.__version__,cuda=torch.version.cuda)
        try:
            import sgl_kernel.flash_attn as fa
        except ImportError as error:
            report.update(status='unsupported',reason=str(error));save();return 3
        wrapper=Path(inspect.getfile(fa));report['installed_wrapper']=dict(path=str(wrapper),sha256=sha(wrapper))
        assert sha(wrapper)==WRAPPER_SHA,'Installed wrapper differs from reviewed source'
        (args.out/'installed_flash_attn.py').write_bytes(wrapper.read_bytes())
        binary=Path(fa.flash_ops.__file__)
        report['native_binary']={'path':str(binary),'sha256':sha(binary),'source_revision_not_proven_by_binary_hash':True}
        if not fa.is_fa3_supported(device):
            report.update(status='unsupported',reason='Installed FA3 capability gate rejected this device');save();return 3
        assert sha(HELPER)==HELPER_SHA,'Helper differs from reviewed source'
        spec=importlib.util.spec_from_file_location('exact_swa_helper',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        assert helper.supports_swa_scheduler_reuse(fa.get_scheduler_metadata)
        torch.manual_seed(9417);torch.cuda.reset_peak_memory_stats()
        # Pairwise coverage, not exhaustive: padded batch/page/window/split/softcap.
        configs=[(1,1,127,0,0.),(4,1,2047,1,0.),(4,16,127,4,0.),(1,16,2047,0,0.),(4,16,2047,1,30.),(1,1,127,4,30.)]
        for batch,page,window,splits,softcap in configs:
            row=dict(batch=batch,page=page,window=window,num_splits=splits,softcap=softcap,checks=0,graph_captures=0)
            report['rows'].append(row);save()
            cap=4096;maxq=8;hq=16;hk=2;dim=64;layers=4
            lens=torch.full((batch,),256,dtype=torch.int32,device='cuda')
            cuq=torch.arange(batch+1,dtype=torch.int32,device='cuda')*maxq
            cuk=torch.arange(batch+1,dtype=torch.int32,device='cuda')*256
            table=torch.arange(batch*(cap//page),dtype=torch.int32,device='cuda').view(batch,cap//page)
            canonical_table=table.clone()
            meta=NS(cache_seqlens_int32=lens,cu_seqlens_q=cuq,cu_seqlens_k=cuk,page_table=table,swa_page_table=table,image_token_ranges=None,swa_spec_metadata=None)
            fb=NS(batch_size=batch,forward_mode=NS(is_target_verify=lambda:True))
            layer=NS(tp_q_head_num=hq,tp_k_head_num=hk,head_dim=dim,v_head_dim=dim,logit_cap=softcap,is_cross_attention=False)
            q=[torch.randn(batch*maxq,hq,dim,device='cuda',dtype=torch.bfloat16)*0.25 for _ in range(layers)]
            k=[torch.randn(batch*cap//page,page,hk,dim,device='cuda',dtype=torch.bfloat16)*0.25 for _ in range(layers)]
            v=[torch.randn_like(x)*0.25 for x in k]
            native_calls=[]
            def factory(**kwargs):
                native_calls.append(tuple(kwargs['window_size']))
                return fa.get_scheduler_metadata(**kwargs)
            factory.__signature__=inspect.signature(fa.get_scheduler_metadata)
            backend=NS(_get_scheduler_metadata=factory,_swa_scheduler_reuse_disabled=False,fa_impl_ver=3,use_mla=False,topk=1,_disable_scheduler_metadata_precompute=False,local_attn_builder=None,_kv_shard_pool=None,is_prefill_aware_swa=False,page_size=page,num_attention_heads=hq,num_kv_heads=hk,head_dim=dim,has_softcap=softcap>0,num_splits=splits)
            def body(kind,debug=False):
                helper.reset_swa_scheduler_cache(backend,fb,allow=True)
                outputs=[];plans=[]
                for li in range(layers):
                    if kind=='fresh':helper.reset_swa_scheduler_cache(backend,fb,allow=True)
                    sched=None
                    if kind!='baseline':
                        sched=helper.get_swa_verify_scheduler_metadata(backend,fb,meta,layer,q[li],k[li],v[li],table,lens,cuq,cuk,maxq,(window,0),causal=True,cp_active=False,use_local_attn=False,use_cascade_attn=False,sinks=None,score_mod=None,rel_bias=None)
                        assert sched is not None
                    outputs.append(fa.flash_attn_with_kvcache(q[li],k[li],v[li],cache_seqlens=lens,page_table=table,cu_seqlens_q=cuq,cu_seqlens_k_new=cuk,max_seqlen_q=maxq,causal=True,window_size=(window,0),softcap=softcap,num_splits=splits,scheduler_metadata=sched,ver=3))
                    if debug and sched is not None:plans.append(sched.clone())
                return outputs,plans
            def same(xs,ys):
                for x,y in zip(xs,ys):
                    assert torch.equal(x.view(torch.int16),y.view(torch.int16)),'Bitwise attention output mismatch'
                    assert torch.isfinite(x).all(),'Nonfinite attention output'
                    row['checks']+=1
            def live_plan(plan):
                # Native local layout: [splits,mblocks,headswizzle], each padded to4,
                # followed by optional semaphore. Padding lanes are uninitialized.
                rounded=(batch+3)//4*4
                vals=[plan[i*rounded:i*rounded+batch] for i in range(3)]
                if plan.numel()>3*rounded:
                    assert plan.numel()==3*rounded+1
                    assert plan[-1].item()==0,'Semaphore not reset after layer'
                return vals
            # Eager numerical and each-layer semaphore proof for baseline/fresh/reuse.
            baseline,_=body('baseline');fresh,freshplans=body('fresh',True);reuse,reuseplans=body('reuse',True)
            torch.cuda.synchronize();same(baseline,fresh);same(baseline,reuse)
            for old,new in zip(freshplans,reuseplans):
                for x,y in zip(live_plan(old),live_plan(new)):
                    assert torch.equal(x,y),'Live scheduler vectors differ';row['checks']+=1
            del baseline,fresh,reuse,freshplans,reuseplans
            graphs={};outputs={};plans={}
            for name,kind in [('A1','baseline'),('B1','reuse'),('A2','baseline'),('B2','reuse')]:
                for _ in range(2):body(kind)
                torch.cuda.synchronize()
                helper.reset_swa_scheduler_cache(backend,fb,allow=False)
                before=len(native_calls);graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out,_=body(kind)
                assert len(native_calls)-before==(0 if kind=='baseline' else 1),'Capture omitted or duplicated scheduler operation'
                graphs[name]=graph;outputs[name]=out
                if kind=='reuse':plans[name]=next(iter(backend._swa_verify_scheduler_cache.values()))
                row['graph_captures']+=1
            assert plans['B1'].data_ptr()!=plans['B2'].data_ptr(),'Separate graphs share scheduler allocation'
            for step in range(6):
                newlens=[([8,129,2048,4096,127,512][step]+i*29-1)%cap+1 for i in range(batch)]
                if batch>1 and step==2:newlens[-1]=0  # padded inactive request
                lens.copy_(torch.tensor(newlens,dtype=torch.int32,device='cuda'))
                cuk[0].zero_();torch.cumsum(lens,dim=0,out=cuk[1:])
                table.copy_(canonical_table.roll(step+1,dims=1))
                for li in range(layers):
                    q[li].normal_(0,0.25);k[li].normal_(0,0.25);v[li].normal_(0,0.25)
                helper.reset_swa_scheduler_cache(backend,fb,allow=False)
                before=len(native_calls)
                for graph in graphs.values():graph.replay()
                torch.cuda.synchronize();assert len(native_calls)==before
                for name in ('A2','B1','B2'):same(outputs['A1'],outputs[name])
                for plan in plans.values():live_plan(plan)
                # Compare graph-owned plan with fresh native plan for current lengths.
                _,freshplans=body('fresh',True);torch.cuda.synchronize()
                for plan in plans.values():
                    for x,y in zip(live_plan(plan),live_plan(freshplans[0])):
                        assert torch.equal(x,y),'Replay plan stale after length update';row['checks']+=1
                del freshplans
            # Paired event times cover 4 consecutive attention layers and scheduler,
            # not full serving, CPU dispatch, or decode latency. Keep all controls.
            timings={name:[] for name in graphs}
            for _ in range(3):
                for graph in graphs.values():graph.replay()
            torch.cuda.synchronize()
            for sample in range(args.samples):
                order=('A1','B1','A2','B2') if sample%2==0 else ('B2','A2','B1','A1')
                for name in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();graphs[name].replay();end.record();end.synchronize()
                    timings[name].append(start.elapsed_time(end)*1000)
            med={name:statistics.median(values) for name,values in timings.items()}
            controls={kind:max(med[kind+'1'],med[kind+'2'])/min(med[kind+'1'],med[kind+'2'])-1 for kind in ('A','B')}
            row.update(passed=True,timings_us=timings,median_us=med,control_relative_difference=controls,controls_pass=all(x<=0.05 for x in controls.values()),speedup=statistics.median(timings['A1']+timings['A2'])/statistics.median(timings['B1']+timings['B2']))
            save();print(json.dumps(row),flush=True)
            graphs.clear();outputs.clear();plans.clear();backend._swa_verify_scheduler_cache=None
        report.update(status='passed',passed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),finished=time.time());save();return 0
    except Exception as error:
        message=str(error)
        unsupported=any(token in message.lower() for token in ('no kernel image','invalid device function','cannot import fa3','can not import fa3'))
        report.update(status='unsupported' if unsupported else 'failed',error=repr(error),traceback=traceback.format_exc(),finished=time.time());save()
        print(report['traceback'],flush=True);return 3 if unsupported else 1

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--samples',type=int,default=12)
    args=pa.parse_args();assert args.samples==12
    raise SystemExit(main(args))
