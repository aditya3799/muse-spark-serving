"""Build an isolated, cumulative pinned-source eager-tree research patch."""
import ast
import difflib
import hashlib
import json
import os
import subprocess
from pathlib import Path

H=Path(__file__).resolve().parent;R=H.parent;D=H/"fixed_tree";D.mkdir(exist_ok=True)
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
par=(R/"release_2026-11-18/candidate.diff").read_bytes()
assert hashlib.sha256(par).hexdigest()=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
repo=R/"sglang"
env=dict(os.environ,GIT_INDEX_FILE=str(D/"build.index"))
def git(*args,input=None):
    return subprocess.check_output(["git","-C",str(repo),*args],input=input,env=env)
git("read-tree",PIN)
git("apply","--cached","--whitespace=nowarn","-",input=par)
S="python/sglang/srt/speculative/";K=R/"research_2026-11-18_rl/lattice_tree"
def get(path):return git("show",":"+path).decode()
def put(path,code):
    ast.parse(code)
    obj=git("hash-object","-w","--stdin",input=code.encode()).decode().strip()
    git("update-index","--add","--cacheinfo","100644,"+obj+","+path)
def rep(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new)
path=S+"fixed_tree_kernels.py"
code=""
for src in ("fused_tree.py","tree_expand.py","tree_pages.py","tree_commit.py"):
    lines=(K/src).read_text().splitlines(True)
    mod=ast.parse("".join(lines))
    for f in mod.body:
        if isinstance(f,ast.FunctionDef) and f.name!="build":
            i=f.decorator_list[0].lineno if f.decorator_list else f.lineno
            code+="\n"+"".join(lines[i-1:f.end_lineno])+"\n"
code+="\n"+(H/"tree_compact_pages.py").read_text(encoding="utf-8-sig")+"\n"
put(path,"import triton\nimport triton.language as tl\n"+code)
adapter=(H/"fixed_tree_adapter.py").read_text(encoding="utf-8-sig")
adapter=rep(adapter,"from fused_tree import _grow\nfrom tree_expand import _expand\nfrom tree_pages import fill_tree_suffix\nfrom tree_commit import _commit",
    "from sglang.srt.speculative.fixed_tree_kernels import _grow, _expand, fill_tree_suffix, _commit")
adapter=adapter.replace("from tree_compact_pages import fill_compact_swa","from sglang.srt.speculative.fixed_tree_kernels import fill_compact_swa")
put(S+"fixed_tree_adapter.py",adapter)
put(S+"fixed_tree_dispatch.py",(H/"tree_dispatch.py").read_text(encoding="utf-8-sig"))
path=S+"dflash_info.py";old=get(path)
put(path,rep(old,"    topk: int = 1","    topk: int = 1\n    tree_step: object = None"))
path="python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py";old=get(path)
put(path,rep(old,"    def can_run_graph(self, forward_batch: ForwardBatch):","""    def can_run_graph(self, forward_batch: ForwardBatch):
        if getattr(forward_batch.spec_info, "tree_step", None) is not None:
            return False"""))
path="python/sglang/srt/layers/attention/flashattention_backend.py";old=get(path)
p=old.index("    def forward_extend(");e=old.index("    def forward_decode(",p)
region=rep(old[p:e],"        # Use precomputed metadata across all layers","""        tree = getattr(forward_batch.spec_info, "tree_step", None)
        if tree is not None:
            if (return_lse or cp_active or score_mod is not None or sinks is not None
                    or rel_bias is not None or aux_tensors is not None):
                raise RuntimeError("Unsupported per-layer tree attention modifier")
            out = getattr(forward_batch, "_attn_output", None)
            if out is not None:
                out = out.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            return tree.attention(self, layer, q, flash_attn_with_kvcache, out).view(
                -1,layer.tp_q_head_num * layer.v_head_dim
            )

        # Use precomputed metadata across all layers""")
put(path,old[:p]+region+old[e:])
path=S+"dflash_worker_v2.py";old=get(path)
old=rep(old,"self, *, draft_model, block_size, max_bs, device, sampling_enabled: bool",
            "self, *, draft_model, block_size, max_bs, device, sampling_enabled: bool, tree_enabled=False")
old=rep(old,"            return _SelectorDraftSampler(", "            from sglang.srt.speculative.fixed_tree_dispatch import tree_runtime_compatible\n            return _SelectorDraftSampler(\n                tree_enabled=tree_runtime_compatible(self),")
old=rep(old,"        self.uniforms = torch.empty((max_bs, gamma), dtype=torch.float32, device=device)",
"""        self.tree_scores = (torch.empty((max_bs,gamma,top_k,top_k),dtype=torch.float32,device=device)
            if tree_enabled and gamma==7 and top_k==16 else None)
        self.uniforms = torch.empty((max_bs, gamma), dtype=torch.float32, device=device)""")
needle="        candidate_ids, scores = _selector_lattice(self.draft_model, hs, block_ids[:, 0])"
old=rep(old,needle,needle+"""
        if self.tree_scores is not None:
            self.tree_scores[:bs].copy_(scores)""")
needle="        verify_input_ids = draft_tokens.reshape(-1)"
old=rep(old,needle,"""        from sglang.srt.speculative.fixed_tree_dispatch import select_tree, accept_tree
        tree_step = select_tree(self,batch,prefix_lens,verify_out_cache_loc_2d,
                                block_ids[:,0],draft_out.can_run_graph)
        if tree_step is not None:
            draft_tokens,positions = tree_step.tokens,tree_step.positions
"""+needle)
old=rep(old,"            draft_token=verify_input_ids,","            draft_token=verify_input_ids,\n            tree_step=tree_step,")
# No kernel updates plan after preparing graph metadata. The runner will admit only eager verify.
start=old.index("        candidates = draft_tokens\n")
end=old.index("        sampling_mask_capture =",start)
orig=old[start:end].rstrip()
new="""        if tree_step is not None:
            accept_len,commit_lens,bonus,out_tokens,new_seq_lens=accept_tree(
                self,tree_step,target_predict,logits_output.next_token_logits,batch,
                prefix_lens,verify_out_cache_loc_2d)
        else:
"""+ "\n".join("    "+line for line in orig.splitlines())+"\n\n"
old=old[:start]+new+old[end:]
needle="        if on_publish is not None:\n            on_publish(new_seq_lens)"
old=rep(old,needle,"        if tree_step is None and on_publish is not None:\n            on_publish(new_seq_lens)")
p=old.index("        cache_loc = self._draft_write_ids(kv_loc_plan)",old.index("# Consume in this step:"))
e=old.index("        # Avoid copying large hidden-state buffers",p)
old=old[:p]+"""        if tree_step is not None:
            tree_step.append_and_publish(self,kv_loc_plan,hidden.flatten(0,1),
                                         on_publish,torch.cuda.Event())
        else:
"""+ "\n".join("    "+s for s in old[p:e].rstrip().splitlines())+"\n\n"+old[e:]
put(path,old)
out=git("diff","--cached","--no-color","--no-ext-diff",PIN)
(D/"candidate.diff").write_bytes(out)
git("read-tree",PIN)
git("apply","--cached","--check","--whitespace=nowarn","-",input=out)
meta=dict(parent_sha256=hashlib.sha256(par).hexdigest(),sha256=hashlib.sha256(out).hexdigest(),
          bytes=len(out),scope="Separate eager fixed-eight target verification, static captured draft selector. No native FA3 or full integration GPU execution yet.")
(D/"patch.json").write_text(json.dumps(meta,indent=2))
print(json.dumps(meta))

