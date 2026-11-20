import argparse,ast,hashlib,importlib.util,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from target_argmax_capture import TargetArgmaxCapture

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'RULES.md').exists())
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
if torch.cuda.mem_get_info()[0] < 768*2**20:raise RuntimeError('reserve: keep TTS running')
softpath=ROOT/'sglang/python/sglang/kernels/ops/sampling/softcap_row_argmax.py'
spec=importlib.util.spec_from_file_location('pinned_softcap',softpath);soft=importlib.util.module_from_spec(spec);spec.loader.exec_module(soft)
lppath=ROOT/'sglang/python/sglang/srt/layers/logits_processor.py'
tree=ast.parse(lppath.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LogitsProcessor')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='deferred_argmax')
method.decorator_list=[]
namespace={'torch':torch,'softcap_row_argmax':soft.softcap_row_argmax}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(lppath),'exec'),namespace)
checks=[]
class Group:
    rank_in_group=0
    world_size=8
    def all_gather_into_tensor(self,output,local):
        # Deliberate single-GPU emulation of eight rank outputs. No timing claim.
        n=local.shape[0];view=output.view(8,n,2);view[0].copy_(local)
        for rank in range(1,8):
            view[rank].copy_(soft.softcap_row_argmax(self.logits[rank],30.0,self.valids[rank],rank*self.logits.shape[-1]))
class LP:
    final_logit_softcapping=30.0
    vocab_size=202048
    deferred_argmax=namespace['deferred_argmax']
    def is_deferred_shard(self,x):return x.shape[-1]<self.vocab_size

group=Group();namespace['get_parallel']=lambda:SimpleNamespace(tp_group=group)
runner=SimpleNamespace(captured_req_width=8)
fb=SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda:True))
lp=LP();tail=TargetArgmaxCapture(lp,max_rows=256,width=8,device='cuda')
# Native per-rank vocab extent. No target/draft model loaded.
for rows in [8,24,32,128,256]:
    group.logits=torch.randn(8,rows,25256,dtype=torch.float32,device='cuda')
    group.valids=[25256]*8
    result=SimpleNamespace(next_token_logits=group.logits[0])
    for _ in range(3):tail.capture_hook(runner,result,fb,rows)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):tail.capture_hook(runner,result,fb,rows)
    for kind in ['random','tied_ranks','softcap_ties','signed_zero','cancellation','changed_random']:
        if kind=='random':group.logits.normal_()
        elif kind=='tied_ranks':group.logits.zero_();group.logits[:, :, 19]=2
        elif kind=='softcap_ties':group.logits.fill_(-100);group.logits[:,:,31]=10000;group.logits[:,:,3]=10001
        elif kind=='signed_zero':group.logits.zero_();group.logits[1::2].neg_()
        elif kind=='cancellation':group.logits.fill_(-1e20);group.logits[6,:,18]=1e20
        else:group.logits.uniform_(-40,40)
        reference=lp.deferred_argmax(result.next_token_logits)
        graph.replay()
        ids=tail.consume(runner=runner,raw_rows=rows,captured_rows=rows,width=8,
                         graph_used=True,can_skip_verify_logits=True,generation=tail.generation)
        assert ids is not None and torch.equal(ids,reference),(rows,kind)
        checks.append({'rows':rows,'case':kind,'same_existing_deferred_argmax':True})
    base=dict(runner=runner,raw_rows=rows,captured_rows=rows,width=8,graph_used=True,
              can_skip_verify_logits=True,generation=tail.generation)
    for key,value in [('graph_used',False),('can_skip_verify_logits',False),('raw_rows',rows-1),('width',4),('generation',999),('runner',SimpleNamespace(captured_req_width=8))]:
        args=dict(base);args[key]=value;assert tail.consume(**args) is None
    # Consume view is overwritten by next replay: explicit same-stream lifetime.
    group.logits.fill_(-20);group.logits[3,:,7]=20;graph.replay()
    assert torch.equal(tail.consume(**base),lp.deferred_argmax(result.next_token_logits))
    del graph,result
    group.logits=None
    torch.cuda.empty_cache()
# Mutation and recapture invalidation must refuse stale output metadata.
base=dict(runner=runner,raw_rows=256,captured_rows=256,width=8,graph_used=True,
          can_skip_verify_logits=True,generation=tail.generation)
original=tail.output
replacement=torch.empty_like(original);tail.output=replacement
assert tail.consume(**base) is None
tail.output=original;tail.invalidate();assert tail.consume(**base) is None
# Actual fixed vocab-shard method and capture hook are tested; collective is emulated.
report={'status':'PASS_LOCAL_CAPTURE_WITH_EMULATED_TP8_ONLY','checks':checks,
        'negative_admission_checks':32,'native_local_vocab':25256,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [lppath,softpath,HERE/'target_argmax_capture.py']},
        'limitations':['No real TP collective,NCCL,NVLink,H200 or full-model validation',
                       'No component or serving speedup inferred from collective emulation',
                       'Non-greedy graph still computes extra argmax; separate variant is future work',
                       'Exact-bucket only; no DP/CP/PP/LoRA/PDMux/concurrent graph execution',
                       'Preserves existing softcap argmax semantics; does not repair its NaN behavior']}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('checks','source_hashes')}))
