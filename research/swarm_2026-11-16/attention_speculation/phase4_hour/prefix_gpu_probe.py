"""Stock greedy-accept + prefix pack diagnostic; no neural inference claim."""
import argparse,hashlib,importlib.util,json,random,time,traceback
from pathlib import Path
import torch
from prefix_boundary import pack_prefix_into
from prefix_protocol import accept_prefix
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
STOCK=ROOT/'sglang/python/sglang/kernels/ops/speculative/dflash.py'
PIN='b4f181c9013915cd64e93348e5d0bdaef3032f8e2826963083c682c85f0e2b8b'
assert hashlib.sha256(STOCK.read_bytes()).hexdigest()==PIN
spec=importlib.util.spec_from_file_location('pinned_dflash_ops',STOCK)
stock=importlib.util.module_from_spec(spec);spec.loader.exec_module(stock)
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);args=pa.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
report={'status':'running','scope':'Integer pack/stock acceptance only; no target M-shape arithmetic, neural draft, full serving or speedup evidence.',
        'stock_sha256':PIN,'cases':[]}
def save():
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
rng=random.Random(88041)
def populate(bs,k,generation,full,predict,prefix):
    tokens=[];positions=[];targetids=[];draftids=[];top=[];lens=[];expected=[]
    for b in range(bs):
        row=[rng.randrange(32000) for _ in range(8)]
        # Includes all accepted, first rejected, and interior rejected cases.
        accepted=(b+generation) % k
        pred=[row[i+1] if i<accepted else (row[min(i+1,7)]+901+i)%32000 for i in range(k)]
        if accepted==k-1: pred[-1]=32700+b
        base=31+generation*7+b*19
        tokens.append(row);positions.append(list(range(base,base+8)))
        targetids.append([1000+b*256+base+i for i in range(8)])
        draftids.append([90000+b*512+base*2+i for i in range(8)])
        top.append(pred);lens.append(base);expected.append(accept_prefix(row[:k],pred,base))
    host=[tokens,positions,targetids,draftids]
    for x,data in zip(full,host):x.copy_(torch.tensor(data,dtype=torch.int64))
    predict.copy_(torch.tensor(top,dtype=torch.int64));prefix.copy_(torch.tensor(lens,dtype=torch.int32))
    return host,top,expected
try:
    report['device']=torch.cuda.get_device_name();report['capability']=torch.cuda.get_device_capability()
    for k in (1,2,4,6,8):
        for bs in (1,4,16,32):
            full=[torch.empty((bs,8),dtype=torch.int64,device='cuda') for _ in range(4)]
            packed=[torch.empty((bs,k),dtype=torch.int64,device='cuda') for _ in range(4)]
            predict=torch.empty((bs,k),dtype=torch.int64,device='cuda')
            prefix=torch.empty(bs,dtype=torch.int32,device='cuda')
            accepted=torch.empty_like(prefix);committed=torch.empty_like(prefix);newlens=torch.empty_like(prefix)
            bonus=torch.empty(bs,dtype=torch.int64,device='cuda')
            output=torch.full((bs,8),-991,dtype=torch.int64,device='cuda')
            def invoke():
                pack_prefix_into(full,packed,k)
                stock._compute_dflash_accept_bonus_triton_unchecked(packed[0],predict,accepted,committed,bonus,output,prefix,newlens)
            def check(host,top,expected,mode):
                for src,dst,data in zip(full,packed,host):
                    assert src.cpu().tolist()==data,'full8 storage mutated'
                    assert dst.cpu().tolist()==[row[:k] for row in data],'prefix pack wrong'
                assert accepted.cpu().tolist()==[x.accepted for x in expected]
                assert committed.cpu().tolist()==[x.committed for x in expected]
                assert bonus.cpu().tolist()==[x.bonus for x in expected]
                assert newlens.cpu().tolist()==[x.new_seq_len for x in expected]
                out=output.cpu().tolist()
                for b,c in enumerate(expected):
                    assert out[b][:c.committed]==list(c.emitted)
                    assert out[b][k:]==[-991]*(8-k),'output tail canary overwritten'
                    targetrow=host[0][b][1:k]+[0];targetrow[c.accepted]=c.bonus
                    assert out[b][:k]==targetrow,'stock inactive token semantics changed'
                report['cases'].append({'batch':bs,'verify_width':k,'mode':mode,'status':'PASS'})
                save()
            host,top,expected=populate(bs,k,0,full,predict,prefix)
            invoke();check(host,top,expected,'eager')
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):invoke()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):invoke()
            for generation in (1,2,7):
                host,top,expected=populate(bs,k,generation,full,predict,prefix)
                graph.replay();check(host,top,expected,'graph_changed_inputs_'+str(generation))
    report.update(status='PASS',checks=len(report['cases']),captures=20,changed_input_replays=60)
except Exception as exc:
    report.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());save();raise
save();print(json.dumps(report))
