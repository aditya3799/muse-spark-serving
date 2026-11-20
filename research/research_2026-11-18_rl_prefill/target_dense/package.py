"""Package group-8 activation and dynamically qualified target projections against the clean SGLang commit."""
import ast
import difflib
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
group=HERE.parent/"candidate.diff"
assert hashlib.sha256(group.read_bytes()).hexdigest()=="33a0a65719afafe691e5c685a7a32ead3194112fc18ce8434f3b31880ce9a721"
prefix="python/sglang/"
runner=prefix+"srt/model_executor/runner/prefill_cuda_graph_runner.py"
model=prefix+"srt/models/muse_spark.py"
helpers=prefix+"srt/layers/communication/muse_spark_gemm_ag.py"
adapter=prefix+"srt/layers/communication/muse_target_projection.py"
native=prefix+"srt/layers/communication/muse_target_projection.cc"

def kernel():
    frozen=HERE/"muse_target_projection.cc"
    if frozen.exists():
        return frozen.read_text(encoding="utf-8")
    p=(ROOT/"research_2026-11-18_rl/lt_probe.cc").read_bytes()
    s=(ROOT/"research_2026-11-18_lt_audit/sweep.cc").read_bytes()
    assert hashlib.sha256(p).hexdigest()=="caaeaf026ec51bd5eb94c356ed770931a1a2f68bfbdfc67e24e3aa1270440fe1"
    assert hashlib.sha256(s).hexdigest() in ("81bbe151e46d533bfd1d5973264e2f05220247a1e550c32cac68aaa2b7ebe321","8832d7b0627a7c4f0a662034dfd2fe1fe7ffef320b89a3c3ec2a55c81cd000d3")
    p=p.decode("utf-8-sig");s=s.decode("utf-8-sig").replace("\r\n","\n")
    # Exclude a later research-only negative max_split path; this adapter passes +4.
    s=re.sub(r"    if \(max_split < 0 && !seeds.empty\(\)\) \{.*?(?=    for\(int split=1;)", "", s, flags=re.S)
    for dest in ("output","parent_output"):
        mark="&p->"+dest+", CUDA_R_16BF"
        assert p.count(mark)==1
        p=p.replace(mark,mark.replace("16BF","32F"))
    mark="&p->output,CUDA_R_16BF"
    assert s.count(mark)==1
    return "\n".join(line.rstrip() for line in s.replace(mark,mark.replace("16BF","32F")).replace('#include "../research_2026-11-18_rl/lt_probe.cc"',p).splitlines()).rstrip()+"\n"

def checked_replace(source,before,after,n=1):
    assert source.count(before)==n,before
    return source.replace(before,after)

def git(tree,*args,input=None):
    proc = subprocess.run(["git","-c","core.autocrlf=false","-C",str(tree),*args],input=input,capture_output=True)
    if proc.returncode: raise RuntimeError(proc.stderr.decode(errors="replace"))
    return proc.stdout

warm="""
    @torch.inference_mode()
    def prepare_target_projection_plans(self, num_tokens: int) -> None:
        parallel = get_parallel()
        if (
            parallel.tp_size not in (2, 4, 8)
            or num_tokens < 512
            or num_tokens % parallel.tp_size
            or parallel.attn_dp_size != 1
            or parallel.attn_cp_size != 1
            or envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability() != (9, 0)
        ):
            return
        from sglang.srt.layers.communication.muse_target_projection import prepare

        layouts = set()
        for layer in self.get_layers():
            if not layer.is_moe_layer or not getattr(layer.mlp.experts, "use_humming", False):
                continue
            for proj in (layer.mlp.pre_expert_proj, layer.mlp.post_expert_proj):
                if proj is None or tuple(proj.weight.shape) in layouts:
                    continue
                layouts.add(tuple(proj.weight.shape))
                prepare(num_tokens // parallel.tp_size, proj.weight)

"""
with tempfile.TemporaryDirectory() as temp:
    tree=Path(temp);git(tree,"init","-q")
    sections=group.read_bytes()
    names=set(re.findall(r"^diff --git a/(\S+) b/",sections.decode(),re.M))|{runner,helpers,model,adapter,native}
    for name in names:
        result=subprocess.run(["git","-C",str(ROOT/"sglang"),"show",PIN+":"+name],capture_output=True)
        if result.returncode==0:
            path=tree/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(result.stdout)
    git(tree,"add","-A")
    git(tree,"-c","user.email=local@research.invalid","-c","user.name=research","commit","-qm","pristine selected files")
    base=git(tree,"rev-parse","HEAD").decode().strip()
    git(tree,"apply","--whitespace=error","-",input=(ROOT/"release_2026-11-18/candidate.diff").read_bytes())
    git(tree,"add","-A");git(tree,"-c","user.email=local@research.invalid","-c","user.name=research","commit","-qm","scored")
    scored=git(tree,"rev-parse","HEAD").decode().strip()
    git(tree,"apply","--whitespace=error","-",input=(HERE.parent/"delta.diff").read_bytes())
    mods={}
    v=(tree/model).read_text(encoding="utf-8")
    before="    def get_attention_sliding_window_size(self) -> Optional[int]:"
    v=checked_replace(v,before,warm+before)
    for a,b in (("get_parallel().tp_group, output, self.post_expert_proj.weight","output"),("get_parallel().tp_group, hidden_states, self.pre_expert_proj.weight","hidden_states")):
        line="prefill_row_projection(\n                    "+a+"\n                )"
        v=checked_replace(v,line,line.replace(a,a+',\n                    qualify_vendor=getattr(self.experts, "use_humming", False),'))
    mods[model]=v
    v=(tree/runner).read_text(encoding="utf-8")
    v=checked_replace(v,"        # Recapture can follow decode replay, which may change model-owned storage.",
        '        prepare_projections = getattr(self.model_runner.model, "prepare_target_projection_plans", None)\n        if prepare_projections is not None:\n            prepare_projections(num_tokens)\n        # Recapture can follow decode replay, which may change model-owned storage.')
    mods[runner]=v
    v=(tree/helpers).read_text(encoding="utf-8")
    v=checked_replace(v,"def prefill_row_projection(group, x, weight):","def prefill_row_projection(group, x, weight, *, qualify_vendor=False):")
    v=checked_replace(v,"    local_fp32 = torch.mm(\n        x[start : start + m_local], weight.t(), out_dtype=torch.float32\n    )",
        '    rows = x[start : start + m_local]\n    if qualify_vendor:\n        from sglang.srt.layers.communication.muse_target_projection import qualified_target_row_mm\n\n        local_fp32 = qualified_target_row_mm(rows, weight)\n    else:\n        local_fp32 = torch.mm(rows, weight.t(), out_dtype=torch.float32)')
    mods[helpers]=v
    mods[adapter]=(HERE/"target_projection.py").read_text(encoding="utf-8-sig").rstrip()+"\n"
    mods[native]=kernel()
    for name,v in mods.items():
        (tree/name).parent.mkdir(parents=True,exist_ok=True)
        (tree/name).write_bytes(v.encode())
        if name.endswith(".py"):ast.parse(v)
    git(tree,"add","-A")
    for parent,name in ((base,"candidate.diff"),(scored,"delta.diff")):
        patch=git(tree,"diff","--cached","--no-ext-diff","--binary",parent)
        (HERE/name).write_bytes(patch)
    git(tree,"reset","--hard",base)
    git(tree,"clean","-fd")
    git(tree,"apply","--check","--whitespace=error","-",input=(HERE/"candidate.diff").read_bytes())
    git(tree,"apply","--whitespace=error","-",input=(HERE/"candidate.diff").read_bytes())
    for name,v in mods.items(): assert (tree/name).read_bytes()==v.encode()
    git(tree,"reset","--hard",scored)
    git(tree,"clean","-fd")
    git(tree,"apply","--check","--whitespace=error","-",input=(HERE/"delta.diff").read_bytes())
    git(tree,"apply","--whitespace=error","-",input=(HERE/"delta.diff").read_bytes())
    for name,v in mods.items(): assert (tree/name).read_bytes()==v.encode()
    (HERE/"muse_target_projection.cc").write_text(mods[native],encoding="utf-8")
    print("Full patch",hashlib.sha256((HERE/"candidate.diff").read_bytes()).hexdigest())

