"""Compose safe row-gather SP and frozen DFlash window omission at the pinned source."""
import ast,difflib,hashlib,json,os,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;SP=H.parent;ROOT=SP.parents[1];REPO=ROOT/'sglang'
EARLY=ROOT/'research_2026-11-18_window_repair'; PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
def digest(x):return hashlib.sha256(x.read_bytes()).hexdigest()
assert digest(EARLY/'early_candidate.diff')=="0238159e2bdbeb5e4e09abc9772d903af0c014ef17cf3830409f062d7dae5182"
assert digest(SP/'candidate.diff')=="431c74e2476474d3ab6e4f3286c9086185add3ffd4f646737727693d936c7eb5"
env=dict(os.environ,GIT_INDEX_FILE=str(H/'integration.index'))
def git(*argv,**kw):return subprocess.check_output(['git','-C',str(REPO),*argv],env=env,**kw)
def delta(path,a,b):return ''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),'a/'+path,'b/'+path)).encode()
def replace(text,a,b):
    assert text.count(a)==1,a
    return text.replace(a,b)
def update(path,new):
    prev=git('show',':'+path).decode()
    ast.parse(new)
    git('apply','--cached','--whitespace=nowarn','-',input=delta(path,prev,new))
    (H/Path(path).name).write_bytes(new.encode())
pieces=ast.parse((EARLY/'build_early.py').read_text())
muse_changes=next(ast.literal_eval(node.value.args[1]) for node in pieces.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='modify' and getattr(node.value.args[0],'id',None)=='muse')
git('read-tree',PIN);git('apply','--cached',str(ROOT/'release_2026-11-18/candidate.diff'))
paths=re.findall(r'^diff --git a/(\S+) b/',(SP/'delta.diff').read_text(),re.M)
baseline={p:git('show',':'+p).decode() for p in paths}
git('read-tree',PIN);git('apply','--cached',str(EARLY/'early_candidate.diff'))
for p in paths:
    early=git('show',':'+p).decode(); sp=(SP/Path(p).name).read_bytes().decode()
    if p.endswith('models/muse_spark.py'):
        expected=baseline[p]
        for a,b in muse_changes:
            expected=replace(expected,a,b);sp=replace(sp,a,b)
        assert expected==early
    else: assert early==baseline[p],p
    update(p,sp)
rp='python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py'
r=git('show',':'+rp).decode()
r=replace(r,'            tail_batch = static_forward_batch\n        if not full_path:', '''            tail_batch = static_forward_batch
            if getattr(forward_batch, "dflash_skip_prefill_aux", False):
                # Replayed bodies keep their captured taps. Omission belongs to
                # this request's eager tail, independently of the capture object.
                tail_batch = copy.copy(static_forward_batch)
                tail_batch.dflash_skip_prefill_aux = True
        if not full_path:''')
update(rp,r)
cand=H/'candidate.diff';cand.write_bytes(git('diff','--cached','--no-ext-diff','--binary',PIN))
git('read-tree',PIN);git('apply','--cached','--check',str(cand))
print(len(cand.read_bytes()),digest(cand))

