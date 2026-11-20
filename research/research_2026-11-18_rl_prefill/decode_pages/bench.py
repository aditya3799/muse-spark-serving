"""Compare frozen FA3 replay metadata kernels with changed graph inputs and guarded reads."""
import argparse,hashlib,importlib.util,json,statistics,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,H/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m.normal_decode_set_metadata
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    old=load('parent_metadata');new=load('metadata')
    g=torch.Generator(device='cuda').manual_seed(a.seed);cap=131072;win=2047
    out=dict(agent='prefill',candidate='Window-aware target FA3 page translation',scope='Single per-forward frozen FA3 metadata kernel, rank-local; core/other stages excluded. Skip inactive in production on RTX, exact candidate kernel replayed.',baseline_sha256=hashlib.sha256((ROOT/'release_2026-11-18/candidate.diff').read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((H/'candidate.diff').read_bytes()).hexdigest(),device=torch.cuda.get_device_name(),seed=a.seed,active=True,correct=True,complete=False,bench=dict(instances=1,shapes={}))
    def bufs(b):return (torch.zeros(b,device='cuda',dtype=torch.int32),torch.zeros(b+1,device='cuda',dtype=torch.int32),torch.zeros((b,cap),device='cuda',dtype=torch.int32),torch.zeros((b,cap),device='cuda',dtype=torch.int32))
    def clock(graph):
        st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
        for _ in range(12):graph.replay()
        en.record();en.synchronize();return st.elapsed_time(en)*1000/12
    for batch,context,q in ((1,2048,1),(1,74160,1),(1,74160,8),(4,74160,8),(16,80000,8),(32,80000,1),(32,80000,8),(3,4097,3),(7,79361,13)):
        dev='cuda';scale=2**19
        requests=torch.randint(1,scale,(batch+3,cap),generator=g,device=dev,dtype=torch.int32)
        to_swa=torch.randperm(scale,device=dev,generator=g,dtype=torch.int32)
        idx=torch.arange(batch,device=dev,dtype=torch.int64);lens=torch.empty(batch,device=dev,dtype=torch.int32)
        def refresh():
            idx.copy_(torch.randperm(batch+3,device=dev,generator=g)[:batch])
            lens.copy_(torch.randint(max(q,context//3),context+1,(batch,),device=dev,generator=g,dtype=torch.int32))
        refresh();arrays={key:bufs(batch) for key in ('baseline','candidate','control')};graphs={}
        delta=0 if q==1 else q
        arrays['candidate'][3].fill_(-1431655766)
        poisoned_dead_pages=0
        def do(name):
            l,cu,pt,sw=arrays[name];f=new if name=='candidate' else old
            return f(l,cu,pt,requests,idx,cap,lens,delta,1,sw,to_swa,**({'swa_suffix_tokens':win+q} if name=='candidate' else {}))
        for key in arrays:
            do(key);cg=torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg):do(key)
            graphs[key]=cg
        times={key:[] for key in arrays};case=f'b{batch}_k{context}_q{q}';access=0
        for i in range(10):
            refresh()
            to_swa.copy_(to_swa.roll(1))
            for key in (('baseline','candidate','control') if i%2==0 else ('control','candidate','baseline')):
                times[key].append(clock(graphs[key]))
            bl,bc,bp,bs=arrays['baseline'];cl,cc,cp,cs=arrays['candidate']
            assert torch.equal(bl,cl) and torch.equal(bc,cc)
            cols=torch.arange(cap,device=dev)[None,:];live=cols<bl[:,None];wanted=live & (cols>=bl[:,None]-q-win)
            assert torch.equal(bp[live],cp[live]) and torch.equal(bs[wanted],cs[wanted])
            access+=int(wanted.sum())
            poisoned_dead_pages+=int(((cs==-1431655766) & live & ~wanted).sum())
        out['bench']['shapes'][case]=dict(weight=1,times=times,checked_swa_reads=access,poisoned_unreachable_per_replay_sum=poisoned_dead_pages)
        if context>4096 and not poisoned_dead_pages:
            # A prior short replay can legally repopulate the whole prefix.
            # Replant poison outside timing, then force a long replay.
            arrays['candidate'][3].fill_(-1431655766)
            lens.fill_(context);graphs['baseline'].replay();graphs['candidate'].replay()
            bl,bc,bp,bs=arrays['baseline'];cl,cc,cp,cs=arrays['candidate']
            cols=torch.arange(cap,device=dev)[None,:];live=cols<bl[:,None];wanted=live & (cols>=bl[:,None]-q-win)
            assert torch.equal(bp[live],cp[live]) and torch.equal(bs[wanted],cs[wanted])
            assert ((cs==-1431655766) & live & ~wanted).any().item()
            out['bench']['shapes'][case]['post_timing_replanted_poison']=True
        print(case, {k:round(statistics.median(v),2) for k,v in times.items()},flush=True)
        a.out.write_text(json.dumps(out,indent=2));del graphs,arrays
    out['complete']=True;a.out.write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()

