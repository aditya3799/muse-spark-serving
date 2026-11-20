"""Build a guarded Muse sequence-parallel adapter against the scored Humming package."""
import ast
import difflib
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO=ROOT/"sglang"
BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
SCORED="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
PREFIX="python/sglang/srt/"
MODEL="models/muse_spark.py"
SP="layers/layernorm_sp.py"
HOOK="arg_groups/layernorm_sp_hook.py"
FLAGS="runtime_context.py"
COLL="layers/communication/muse_spark_gemm_ag.py"

def replace(src,a,b):
    assert src.count(a)==1, a
    return src.replace(a,b)

def edit(name,s):
    if name==FLAGS:
        return replace(s,"    enabled: bool = False\n\n\nclass Flags(", "    enabled: bool = False\n    replicated_muse_experts: bool = False\n\n\nclass Flags(")
    if name==HOOK:
        s=replace(s,"        if envs.SGLANG_MOE_NVFP4_DISPATCH.get():","        if cfg.moe_runner_backend == \"humming\":\n            return\n\n        if envs.SGLANG_MOE_NVFP4_DISPATCH.get():")
        s=replace(s,"        elif moe_a2a_backend != \"flashinfer\":","""        elif moe_runner_backend == "humming":
            if moe_a2a_backend != "none":
                raise ValueError("Muse Humming SP uses replicated packed experts and requires --moe-a2a-backend none.")
        elif moe_a2a_backend != "flashinfer":""")
        return replace(s,'        if moe_runner_backend != "flashinfer_trtllm":\n','        if (moe_runner_backend != "flashinfer_trtllm"\n                and (enable_megamoe_for_layernorm_sp or moe_runner_backend != "humming")):\n')
    if name==SP:
        s=replace(s,"\n\ndef layernorm_sp_enabled()",'''
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec

    # The scheduler has published the config here. initialize_moe_config() seeds
    # the backend object later, so consult the resolved execution setting.
    humming = (get_flags().sp.enabled and is_muse_spark_architecture(architectures[0])
               and get_exec().moe.moe_runner_backend == "humming")
    get_flags().sp.replicated_muse_experts = False
    if not humming:
        return
    text_config = getattr(model_config, "hf_text_config", None)
    widths = (getattr(text_config, "hidden_size", 0), getattr(text_config, "moe_hidden_size", 0))
    get_flags().sp.replicated_muse_experts = bool(
        all(w and w >= 1024 and w % 128 == 0 for w in widths)
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (9, 0)
        and get_parallel().attn_cp_size == 1 and get_parallel().moe_dp_size == 1
        and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
    )
    if not get_flags().sp.replicated_muse_experts:
        get_flags().sp.enabled = False


def layernorm_sp_enabled()''')
        s=replace(s,"def runs_sp(forward_mode) -> bool:", "def runs_sp(forward_mode, num_tokens=None, *, returned_rows=None) -> bool:")
        s=replace(s,"    return layernorm_sp_enabled() and forward_mode == ForwardMode.EXTEND","""    if not layernorm_sp_enabled() or forward_mode != ForwardMode.EXTEND:
        return False
    if get_flags().sp.replicated_muse_experts:
        # Match the existing Humming prefill-row projection eligibility.
        world = get_parallel().tp_group.world_size
        if returned_rows is not None:
            # Full graphs invoke the eager tail with the raw count after running
            # a padded body. The runner caps padding at 2x and Humming TP is 4/8.
            if num_tokens is None or returned_rows >= num_tokens:
                return False
            num_tokens = returned_rows * world
        return num_tokens is not None and num_tokens >= 512 and num_tokens % world == 0
    return True""")
        s=replace(s,"    if not runs_sp(forward_mode) or input_ids is None:", "    if input_ids is None or not runs_sp(forward_mode, input_ids.shape[0], returned_rows=hidden_states.shape[0]):")
        return replace(s,"        not (_HAS_TORCH_SYMM_MEM_FUSED or getattr(linear, \"_sp_use_cutlass\", False))","        getattr(linear, \"_sp_use_native_linear\", False)\n        or not (_HAS_TORCH_SYMM_MEM_FUSED or getattr(linear, \"_sp_use_cutlass\", False))")
    if name==COLL:
        return replace(s,'''    local_fp32 = torch.mm(
        x[start : start + m_local], weight.t(), out_dtype=torch.float32
    )
    local = local_fp32.to(x.dtype)
    del local_fp32
    return group.all_gather(local, dim=0)''','''    return group.all_gather(local_prefill_projection(x[start : start + m_local], weight), dim=0)


def local_prefill_projection(rows, weight):
    """One complete row shard with the parent full-K accumulation and BF16 boundary."""
    return torch.mm(rows, weight.t(), out_dtype=torch.float32).to(rows.dtype)''')
    assert name==MODEL
    s=replace(s,"    prefill_row_projection,\n","    local_prefill_projection,\n    prefill_row_projection,\n")
    s=replace(s,"            if self._can_shard_prefill_row_projection(output, self.post_expert_proj.weight):","""            if getattr(get_forward(), "sp_active", False) and getattr(self.experts, "use_humming", False):
                return local_prefill_projection(output, self.post_expert_proj.weight)
            if self._can_shard_prefill_row_projection(output, self.post_expert_proj.weight):""")
    s=replace(s,"            elif self._can_shard_prefill_row_projection(\n","""            elif sp_active and getattr(self.experts, "use_humming", False):
                expert_input = local_prefill_projection(hidden_states, self.pre_expert_proj.weight)
            elif self._can_shard_prefill_row_projection(
""")
    s=replace(s,'        """Return combined source-token shards, before the nonlinear FFN tail."""\n','''        """Return combined source-token shards, before the nonlinear FFN tail."""
        if getattr(self.experts, "use_humming", False):
            group = get_parallel().tp_group
            global_input = group.all_gather(expert_input.contiguous(), dim=0)
            global_scores = group.all_gather(topk_scores.contiguous(), dim=0)
            global_ids = group.all_gather(topk_ids.contiguous(), dim=0)
            partial = self.experts(
                global_input, global_scores, global_ids, self.experts.post_expert_norm,
                pre_expert_norm=self.pre_expert_norm,
            )
            local = torch.empty_like(expert_input)
            group.reduce_scatter_tensor(local, partial.contiguous())
            return local
''')
    s=replace(s,"        if layernorm_sp.layernorm_sp_enabled():\n", """        replicated_experts = any(\n            getattr(layer.mlp.experts, "use_humming", False) for layer in self.layers if layer.is_moe_layer\n        )\n        if layernorm_sp.layernorm_sp_enabled():\n""")
    s=replace(s,"                    linear._sp_use_cutlass = True","""                    if replicated_experts:
                        # Keep the same linear method on Hopper. The CUTLASS SP path requires SM100.
                        linear._sp_use_native_linear = True
                    else:
                        linear._sp_use_cutlass = True""")
    s=replace(s,"            if moe_layers and not getattr(\n",'''            if moe_layers and not replicated_experts and not getattr(
''')
    s=replace(s,"sp_active = layernorm_sp.runs_sp(forward_batch.forward_mode)","sp_active = layernorm_sp.runs_sp(forward_batch.forward_mode, hidden_states.shape[0])")
    s=replace(s,"        if aux_hidden_states is not None and layernorm_sp.runs_sp(\n            forward_batch.forward_mode\n        ):","""        num_tokens = input_ids.shape[0] if input_ids is not None else input_embeds.shape[0]
        if aux_hidden_states is not None and layernorm_sp.runs_sp(\n            forward_batch.forward_mode, num_tokens, returned_rows=hidden_states.shape[0]\n        ):""")
    s=replace(s,"            num_tokens = (\n                input_ids.shape[0] if input_ids is not None else input_embeds.shape[0]\n            )\n","")
    return replace(s,"and layernorm_sp.runs_sp(forward_batch.forward_mode)","and layernorm_sp.runs_sp(forward_batch.forward_mode, num_tokens, returned_rows=hidden_states.shape[0])")

def diff(path,old,new):
    return "diff --git a/{0} b/{0}\n".format(path)+"".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),"a/"+path,"b/"+path))

def version(name,s,variant):
    if name!=MODEL or variant=="local": return s
    if variant=="scratch":
        from scratch_edit import apply
        return apply(s,replace)
    s=replace(s,"            global_scores = group.all_gather(topk_scores.contiguous(), dim=0)\n            global_ids = group.all_gather(topk_ids.contiguous(), dim=0)\n","")
    s=replace(s,"global_input, global_scores, global_ids, self.experts.post_expert_norm,","global_input, topk_scores, topk_ids, self.experts.post_expert_norm,")
    s=replace(s,"            router_logits = (\n                self.gate.forward_split2(hidden_states)","""            router_hidden = hidden_states
            if sp_active and getattr(self.experts, "use_humming", False):
                # Use exactly the scored M,K,N, weight layout and FP32 partials for routing.
                router_hidden = get_parallel().tp_group.all_gather(hidden_states.contiguous(), dim=0)
            router_logits = (
                self.gate.forward_split2(router_hidden)""")
    s=replace(s,"            if sp_active:\n                capture_topk_ids =", '            if sp_active and not getattr(self.experts, "use_humming", False):\n                capture_topk_ids =')
    return s

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--variant",choices=("local","gather","scratch"),default="gather")
    variant=parser.parse_args().variant
    output=HERE if variant=="gather" else HERE/("experimental_"+variant)
    output.mkdir(exist_ok=True)
    parent=(ROOT/"release_2026-11-18/candidate.diff").read_bytes()
    assert hashlib.sha256(parent).hexdigest()==SCORED
    original=re.split(r"(?=^diff --git )",parent.decode(),flags=re.M)[1:]
    sections={p.splitlines()[0].split()[2][2:]:p for p in original}
    with tempfile.TemporaryDirectory(dir=HERE) as t:
        tmp=Path(t)
        def apply(b,*flags):
            subprocess.run(["git","-C",t,"apply","--whitespace=error",*flags,"-"],input=b,check=True)
        subprocess.run(["git","-C",t,"init","-q"],check=True)
        base={}
        with_parent={}
        delta=[]
        for name in (HOOK,FLAGS,COLL,SP,MODEL):
            path=PREFIX+name
            b=subprocess.check_output(["git","-C",str(REPO),"show",BASE+":"+path])
            dest=tmp/path
            dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(b)
            base[path]=b
            if path in sections: apply(sections[path].encode())
            prev=dest.read_bytes().decode()
            nxt=version(name,edit(name,prev),variant)
            ast.parse(nxt)
            delta.append(diff(path,prev,nxt))
            (output/Path(name).name).write_bytes(nxt.encode())
            sections[path]=diff(path,b.decode(),nxt)
            with_parent[path]=dest.read_bytes()
        changes="".join(delta).encode(); apply(changes,"--check");apply(changes)
        for path in base:
            assert (tmp/path).read_bytes()==(output/Path(path).name).read_bytes()
            (tmp/path).write_bytes(base[path])
        apply("".join(sections[p] for p in base).encode(),"--check")
        (output/"delta.diff").write_bytes(changes)
        bundle="".join(sections.values()).encode()
        (output/"candidate.diff").write_bytes(bundle)
        print(len(bundle),hashlib.sha256(bundle).hexdigest())
if __name__=="__main__":main()

