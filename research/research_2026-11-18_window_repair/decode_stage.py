"""Scoped addition to the already verified captured direct-output selector."""
from pathlib import Path
import ast,difflib,os,subprocess,hashlib,json,statistics,time
from types import SimpleNamespace as S
import torch
H=Path(__file__).resolve().parent;D=H/"decode_selector";R=H.parent;PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
f="python/sglang/srt/speculative/dflash_worker_v2.py"
old=(D/"dflash_worker_v2_direct_src.py").read_text(encoding="utf8");new=old
where="        self.greedy_mask = torch.ones((max_bs,), dtype=torch.bool, device=device)"
assert new.count(where)==1
new=new.replace(where,where+'''
        self._uniform_greedy_prefix = 0
        self._uniform_greedy_stream = None''')
where='''        if sampling_info is None or not self.sampling_enabled:
            self.temperatures[:bs].fill_(1.0)
            self.greedy_mask[:bs].fill_(True)
            return
        torch.clamp('''
assert new.count(where)==1
new=new.replace(where,'''        stream = torch.cuda.current_stream(self.greedy_mask.device).cuda_stream if self.greedy_mask.is_cuda else None
        if stream != self._uniform_greedy_stream:
            self._uniform_greedy_prefix = 0
            self._uniform_greedy_stream = stream
        if sampling_info is None or not self.sampling_enabled or getattr(sampling_info, "is_all_greedy", False):
            if bs > self._uniform_greedy_prefix:
                start = self._uniform_greedy_prefix
                self.temperatures[start:bs].fill_(1.0)
                self.greedy_mask[start:bs].fill_(True)
                self._uniform_greedy_prefix = bs
            return
        self._uniform_greedy_prefix = 0
        torch.clamp(''')
ast.parse(new)
(D/"dflash_worker_v2_staged_src.py").write_text(new,encoding="utf8")
env={**os.environ,"GIT_INDEX_FILE":str(D/"staged.index")}
def git(*args,**kw):return subprocess.check_output(["git","-C",str(R/"sglang"),*args],env=env,**kw)
git("read-tree",PIN);git("apply","--cached",str(D/"direct_candidate.diff"))
inc="".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),"a/"+f,"b/"+f))
git("apply","--cached","-",input=inc.encode());out=D/"staged_candidate.diff";out.write_bytes(git("diff","--cached","--no-ext-diff","--binary",PIN))
git("read-tree",PIN);git("apply","--check","--cached",str(out))
def sampler(src,enabled=True):
    cl=next(n for n in ast.parse(src).body if getattr(n,"name",None)=="_SelectorDraftSampler")
    scope=dict(torch=torch,resolve_greedy_mask=lambda *,sampling_info,bs,device:(sampling_info.top_ks<=1).view(-1))
    exec(compile(ast.Module(body=[cl],type_ignores=[]),f,"exec"),scope)
    return scope[cl.name](draft_model=S(candidate_selector=S(top_k=16)),block_size=8,max_bs=32,device="cpu",sampling_enabled=enabled)
def fixture(n,mask,g):
    if mask=="absent":return None
    ks=torch.ones(n,dtype=torch.int32)
    if mask=="mixed":ks[::2]=7
    return S(temperatures=torch.rand(n,generator=g)*2,top_ks=ks,is_all_greedy=bool(torch.all(ks<=1)))
def check():
    gen=torch.Generator().manual_seed(172991);tested=0
    for enabled in [False,True]:
        a,b=sampler(old,enabled),sampler(new,enabled)
        for i in range(1000):
            n=int(torch.randint(1,33,(1,),generator=gen))
            info=fixture(n,("absent","greedy","mixed","greedy","greedy")[i%5],gen)
            for w in (a,b):w.stage_sampling_params(bs=n,sampling_info=info)
            assert torch.equal(a.greedy_mask[:n],b.greedy_mask[:n])
            is_sample=~a.greedy_mask[:n]
            assert torch.equal(a.temperatures[:n][is_sample],b.temperatures[:n][is_sample])
            tested+=n
    return tested
if __name__=="__main__":
    result=dict(active_rows=check(),changing_batch_sequences=2000,source="SamplingBatchInfo.from_schedule_batch sets is_all_greedy from the same top_k<=1 predicate as resolve_greedy_mask",sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (D/"staging_cpu.json").write_text(json.dumps(result,indent=2));print(result)

