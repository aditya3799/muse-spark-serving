"""Compare frozen FA3 replay metadata kernels with changed graph inputs and guarded reads."""
import argparse,hashlib,importlib.util,json,statistics,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
def load(name):
    if name=='window_metadata':
        spec=importlib.util.spec_from_file_location(name,H.parent/'metadata.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m.normal_decode_set_metadata
    spec=importlib.util.spec_from_file_location(name,(H.parent if name=='parent_metadata' else H)/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m.normal_decode_set_metadata
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allocator-cap-mib',type=int,default=512);a=p.parse_args()
    free0,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(a.allocator_cap_mib*2**20/total);torch.cuda.reset_peak_memory_stats()
    old=load('parent_metadata');new=load('metadata');window=load('window_metadata')
    g=torch.Generator(device='cuda').manual_seed(a.seed);cap=131072;win=2047
    out=dict(agent='prefill',candidate='Direct FA3 request page table and suffix-only SWA translation',scope='Single FA3 metadata launch. Direct live global request row map and suffix SWA, native row-indirection address equivalence checked separately. Additional FA3 cache_batch_idx load and post-replay scheduler barrier excluded.',baseline_sha256=hashlib.sha256((ROOT/'release_2026-11-18/candidate.diff').read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((H/'candidate.diff').read_bytes()).hexdigest(),device=torch.cuda.get_device_name(),seed=a.seed,active=True,correct=True,complete=False,bench=dict(instances=1,shapes={}))
    out['memory']={'free_after_cuda_context':free0,'total_b':total,'pytorch_cap_mib':a.allocator_cap_mib}
    def bufs(b):return (torch.zeros(b,device='cuda',dtype=torch.int32),torch.zeros(b+1,device='cuda',dtype=torch.int32),torch.zeros((b,cap),device='cuda',dtype=torch.int32),torch.zeros((b,cap),device='cuda',dtype=torch.int32))
    def clock(graph):
        st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
        for _ in range(12):graph.replay()
        en.record();en.synchronize();return st.elapsed_time(en)*1000/12
    out['fixture']={'full_slots':1893608,'swa_slots':1514886,'page_size':1,'req_dtype':'int32','index_dtype':'int64','map_dtype':'int64','swa_mapping':'evicted interior zero, exclusive positive reachable suffix slots, trailing -1 sentinel','layouts':['contiguous','512-token extents','fully fragmented']}
    cases=[(1,2048,1,1),(3,4097,3,512),(7,79361,13,512)]+[(b,74160,q,extent) for b,q in ((1,8),(16,8),(32,1),(32,8)) for extent in (0,512,1)]
    for batch,context,q,extent in cases:
        dev='cuda';scale=1893608;swacap=1514886
        # Reserve disjoint maximum live ranges per request. A 32-request mix
        # fits the scored card's full-token budget even at the maximum lengths.
        maxima=[context if batch==1 else int(context*(.4+.6*r/(batch-1))) for r in range(batch)]
        assert sum(maxima)+batch*q<=scale
        requests=torch.zeros((33,cap),device=dev,dtype=torch.int32)
        slots=torch.arange(1,scale+1,device=dev,dtype=torch.int32)
        if extent:
            blocks=scale//extent;size=blocks*extent
            perm=torch.randperm(blocks,device=dev,generator=g)
            slots[:size]=slots[:size].reshape(blocks,extent)[perm].flatten()
        start=0
        for ri,maxlen in enumerate(maxima):
            length=maxlen+q
            requests[ri+1,:length]=slots[start:start+length];start+=length
        to_swa=torch.zeros(scale+2,device=dev,dtype=torch.int64)
        idx=torch.arange(batch,device=dev,dtype=torch.int64);lens=torch.empty(batch,device=dev,dtype=torch.int32)
        limit=torch.tensor(maxima,device=dev,dtype=torch.int32)
        def refresh():
            idx.copy_(torch.randperm(batch,device=dev,generator=g)+1)
            lens.copy_((limit[idx-1].float()*(.65+.35*torch.rand(batch,device=dev,generator=g))).int())
        refresh();arrays={key:bufs(batch) for key in ('baseline','candidate','window','control')};graphs={}
        ids=torch.empty(batch,device=dev,dtype=torch.int32)
        arrays['candidate'][2].fill_(-302113)
        delta=0 if q==1 else q
        arrays['candidate'][3].fill_(-1431655766)
        poisoned_dead_pages=0
        def do(name):
            l,cu,pt,sw=arrays[name];f=new if name=='candidate' else (window if name=='window' else old)
            return f(l,cu,pt,requests,idx,cap,lens,delta,1,sw,to_swa,**({'swa_suffix_tokens':win+q,'direct_full_indices':ids} if name=='candidate' else ({'swa_suffix_tokens':win+q} if name=='window' else {})))
        def remap_suffix(phase):
            # Match the allocator's zeroed/evicted interior and -1 end sentinel.
            # Assign distinct positive physical SWA slots to every live tail.
            positions=(lens+delta)[:,None]-1-torch.arange(win+q,device=dev)[None,:]
            full=requests[idx[:,None],positions.clamp_min(0).long()][positions>=0].long()
            n=full.numel()
            assert n<=swacap and torch.unique(full).numel()==n and full.min().item()>0
            physical=(torch.arange(n,device=dev,dtype=torch.int64)+phase*7919)%swacap+1
            to_swa.zero_()
            to_swa[full]=physical
            to_swa[-1]=-1
        remap_suffix(0)
        for key in arrays:
            do(key);cg=torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg):do(key)
            graphs[key]=cg
        times={key:[] for key in arrays};case=f'b{batch}_k{context}_q{q}_extent{extent}';access=0
        for i in range(10):
            refresh()
            remap_suffix(i+1)
            for key in (('baseline','candidate','window','control') if i%2==0 else ('control','window','candidate','baseline')):
                times[key].append(clock(graphs[key]))
            bl,bc,bp,bs=arrays['baseline'];cl,cc,cp,cs=arrays['candidate']
            assert torch.equal(bl,cl) and torch.equal(bc,cc)
            cols=torch.arange(cap,device=dev)[None,:];live=cols<bl[:,None];wanted=live & (cols>=bl[:,None]-q-win)
            assert torch.equal(ids,idx.int()) and torch.equal(bp[live],requests[ids.long()][live])
            assert (cp==-302113).all().item() and torch.equal(bs[wanted],cs[wanted])
            wl,wcu,wp,ws=arrays['window']; assert torch.equal(wp[live],bp[live]) and torch.equal(ws[wanted],bs[wanted])
            access+=int(wanted.sum())
            poisoned_dead_pages+=int(((cs==-1431655766) & live & ~wanted).sum())
        out['bench']['shapes'][case]=dict(weight=1,times=times,checked_swa_reads=access,poisoned_unreachable_per_replay_sum=poisoned_dead_pages)
        if context>4096 and not poisoned_dead_pages:
            # A prior short replay can legally repopulate the whole prefix.
            # Replant poison outside timing, then force a long replay.
            arrays['candidate'][3].fill_(-1431655766)
            lens.copy_(limit[idx-1]);remap_suffix(13);graphs['baseline'].replay();graphs['candidate'].replay()
            bl,bc,bp,bs=arrays['baseline'];cl,cc,cp,cs=arrays['candidate']
            cols=torch.arange(cap,device=dev)[None,:];live=cols<bl[:,None];wanted=live & (cols>=bl[:,None]-q-win)
            assert torch.equal(ids,idx.int()) and torch.equal(bp[live],requests[ids.long()][live])
            assert (cp==-302113).all().item() and torch.equal(bs[wanted],cs[wanted])
            assert ((cs==-1431655766) & live & ~wanted).any().item()
            out['bench']['shapes'][case]['post_timing_replanted_poison']=True
        print(case, {k:round(statistics.median(v),2) for k,v in times.items()},flush=True)
        out['memory'].update(peak_pytorch_alloc_b=torch.cuda.max_memory_allocated(),peak_pytorch_reserved_b=torch.cuda.max_memory_reserved(),global_free_b=torch.cuda.mem_get_info()[0]);a.out.write_text(json.dumps(out,indent=2))
        del graphs,arrays,cg,bl,bc,bp,bs,cl,cc,cp,cs,wl,wcu,wp,ws,live,wanted,cols
        if extent:del perm
    # Make the early-publication hazard observable by explicitly ordering a
    # scheduler-side write after metadata staging but before the old read event.
    def lifetime(wait_post):
        primary=torch.cuda.current_stream();writer=torch.cuda.Stream()
        cells=torch.tensor([[1,2],[17,19]],dtype=torch.int32,device='cuda')
        rows=torch.tensor([1,1,0],dtype=torch.int32,device='cuda')
        snapshot=cells[rows.long()].clone()
        prepare=torch.cuda.Event();done=torch.cuda.Event();changed=torch.cuda.Event()
        prepare.record(primary)
        if wait_post:
            observed=cells[rows.long()].clone()
            done.record(primary)
        with torch.cuda.stream(writer):
            writer.wait_event(done if wait_post else prepare);cells.fill_(82);changed.record(writer)
        primary.wait_event(changed)
        if not wait_post:observed=cells[rows.long()].clone()
        return torch.equal(observed,snapshot)
    out['stream_ownership']={'pre_event_preserved':lifetime(False),'post_event_preserved':lifetime(True)}
    assert out['stream_ownership']=={'pre_event_preserved':False,'post_event_preserved':True}
    out['complete']=True;a.out.write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()

