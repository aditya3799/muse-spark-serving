"""Native FA3 BF16 shape-preserving prefix gate, no model inference or compile."""
import argparse,hashlib,inspect,json,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent
PIN='610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac'
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);args=pa.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
report={'status':'running','rows':[],'scope':'Native installed FA3 BF16 only, full8 native query/KV/scheduler geometry, real firstK query/KV rows and zero tail. No H200 FP8/MoE/fullmodel/serving inference. Exact bitwise gate retained.','started':time.time()}
def save():
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
try:
    import torch
    import sgl_kernel.flash_attn as fa
    device=torch.cuda.current_device()
    report['device']={'name':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'torch':torch.__version__,'cuda':torch.version.cuda}
    if not fa.is_fa3_supported(device):
        report.update(status='unsupported',reason='Native capability gate rejected device');save();raise SystemExit(3)
    wrapper=Path(inspect.getfile(fa));digest=hashlib.sha256(wrapper.read_bytes()).hexdigest()
    assert digest==PIN,'Unreviewed installed wrapper'
    (args.out/'installed_flash_attn.py').write_bytes(wrapper.read_bytes())
    binary=Path(fa.flash_ops.__file__)
    report['native']={'wrapper_sha256':digest,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'binary_path':str(binary)}
    torch.manual_seed(8004)
    for splits in (0,):
      for window in ((-1,-1),(2047,0)):
       for bs in (1,4,16,32):
        for width in (1,2,4,6):
            row={'batch':bs,'width':width,'window':window,'num_splits':splits,'page':1,'attention_width':8,'tail':'zero query and K/V for unverified inputs','checks':[]}
            report['rows'].append(row);save()
            try:
                hq,hk,dim,cap=16,2,64,4096
                q8=torch.randn(bs*8,hq,dim,device='cuda',dtype=torch.bfloat16)*.25
                qk=torch.empty_like(q8)
                kc=torch.randn(bs*cap,1,hk,dim,device='cuda',dtype=torch.bfloat16)*.25
                vc=torch.randn_like(kc)*.25
                kpad=torch.empty_like(kc);vpad=torch.empty_like(vc)
                table=torch.arange(bs*cap,device='cuda',dtype=torch.int32).view(bs,cap)
                canonical=table.clone()
                len8=torch.full((bs,),135,device='cuda',dtype=torch.int32)
                lenk=torch.full((bs,),127+width,device='cuda',dtype=torch.int32)
                cu8=torch.arange(bs+1,device='cuda',dtype=torch.int32)*8
                cuk=torch.arange(bs+1,device='cuda',dtype=torch.int32)*width
                def update(step):
                    prefix=[0,127,2047,4088][step]
                    lens=[min(4088,prefix+b*7) for b in range(bs)]
                    full=[x+8 for x in lens];short=[x+width for x in lens]
                    if step==3 and bs>1:full[-1]=short[-1]=0
                    len8.copy_(torch.tensor(full,dtype=torch.int32));lenk.copy_(torch.tensor(short,dtype=torch.int32))
                    table.copy_(canonical.roll(step+1,dims=1))
                    q8.normal_(0,.25);kc.normal_(0,.25);vc.normal_(0,.25)
                    qk.copy_(q8);qk.view(bs,8,hq,dim)[:,width:].zero_()
                    kpad.copy_(kc);vpad.copy_(vc)
                    # page1 lets the table identify each physical tail token.
                    # Both arms keep identical length=prefix+8 and scheduler shape.
                    for b,base in enumerate(lens):
                        tail_ids=table[b,base+width:base+8].to(torch.int64)
                        kpad.index_fill_(0,tail_ids,0);vpad.index_fill_(0,tail_ids,0)
                    return {'prefix_lengths':lens,'full_cache_lens':full,'logical_real_prefix_cache_lens':short,'candidate_native_cache_lens':full}
                def body(full):
                    return fa.flash_attn_with_kvcache(q8 if full else qk,kc if full else kpad,vc if full else vpad,
                        cache_seqlens=len8,page_table=table,
                        cu_seqlens_q=cu8,max_seqlen_q=8,
                        causal=True,window_size=window,num_splits=splits,ver=3)
                def compare(full,short,mode,inputs):
                    a=full.view(bs,8,hq,dim)[:,:width].contiguous()
                    b=short.view(bs,8,hq,dim)[:,:width].contiguous()
                    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                    equal=torch.equal(a.view(torch.int16),b.view(torch.int16))
                    mismatch=int((a.view(torch.int16)!=b.view(torch.int16)).sum().item())
                    row['checks'].append({'mode':mode,'status':'PASS' if finite and equal else 'FAIL','bitwise_mismatches':mismatch,'max_abs':float((a.float()-b.float()).abs().max().item()),'finite':finite,'inputs':inputs})
                    save()
                inputs=update(1);full=body(True);short=body(False);compare(full,short,'eager',inputs)
                # Identical full8 repetition is a deterministic numerical control.
                full2=body(True);assert torch.equal(full.view(torch.int16),full2.view(torch.int16)),'Full8 A/A failed'
                outputs={};graphs={}
                for mode in (True,False):
                    for _ in range(2):body(mode)
                    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):out=body(mode)
                    graphs[mode]=graph;outputs[mode]=out
                for step in (0,2,3):
                    inputs=update(step)
                    graphs[True].replay();graphs[False].replay()
                    compare(outputs[True],outputs[False],'changed_graph_'+str(step),inputs)
                # Negative control: changing actual firstK query content must be
                # observable. This catches accidental self-comparison/stale output.
                qk.zero_();graphs[False].replay()
                neg_a=outputs[True].view(bs,8,hq,dim)[:,:width].contiguous()
                neg_b=outputs[False].view(bs,8,hq,dim)[:,:width].contiguous()
                row['negative_real_query_change_detected']=not torch.equal(neg_a.view(torch.int16),neg_b.view(torch.int16))
                assert row['negative_real_query_change_detected'],'Negative real-query control was not detected'
                del neg_a,neg_b
                row['status']='PASS' if all(x['status']=='PASS' for x in row['checks']) else 'FAIL'
                graphs.clear();outputs.clear()
                del graph,out,full,short,full2,q8,qk,kc,vc,kpad,vpad,table,canonical
            except Exception as e:
                row.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
            save()
    failures=sum(x['status']!='PASS' for x in report['rows'])
    report.update(status='FAIL' if failures else 'PASS',failed_cases=failures,cases=len(report['rows']),checks=sum(len(x['checks']) for x in report['rows']),finished=time.time())
    save();print(json.dumps({'status':report['status'],'cases':report['cases'],'failed_cases':failures,'checks':report['checks']}));raise SystemExit(1 if failures else 0)
except SystemExit:raise
except Exception as e:
    report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc(),finished=time.time());save();raise
