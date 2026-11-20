"""Reorder the same target BF16 cat to the last requested Muse layer in eligible verify graphs."""
import ast,difflib,hashlib,os,subprocess
from pathlib import Path
import overlap_event_build as event
D=event.D;R=event.R;LP=event.LP
MODEL="python/sglang/srt/models/muse_spark.py"
parent_pkg=(R/"release_2026-11-18/candidate.diff").read_bytes()
parts={}
for seg in parent_pkg.split(b"diff --git a/")[1:]:
    file=seg.split(b" b/")[0].decode()
    assert file not in parts
    parts[file]=b"diff --git a/"+seg
assert MODEL in parts
source=(R/"release_2026-11-18/candidate_sources"/MODEL).read_text()
assert source.count("        aux_hidden_states = []\n        for layer_idx, layer in enumerate(self.layers):")==1
source=source.replace("        aux_hidden_states = []\n        for layer_idx, layer in enumerate(self.layers):","""        aux_hidden_states = []
        early_context = None
        context_ready = getattr(forward_batch, "dflash_context_ready_event", None)
        if context_ready is not None:
            forward_batch.dflash_tapped_context = None
        for layer_idx, layer in enumerate(self.layers):""")
a="""                aux_hidden_states.append(stream.to(self.operator_dtype))
        norm_input = stream.to(self.operator_dtype)
        hidden_states = self.norm(norm_input).to(self.operator_dtype)
        if aux_hidden_states:"""
assert source.count(a)==1
source=source.replace(a,"""                aux_hidden_states.append(stream.to(self.operator_dtype))
                if (
                    context_ready is not None and not sp_active
                    and len(aux_hidden_states) == len(self.layers_to_capture)
                ):
                    early_context = pack_aux_hidden_states(aux_hidden_states)
                    aux_hidden_states.clear()
                    context_ready.record()
                    forward_batch.dflash_tapped_context = early_context
        norm_input = stream.to(self.operator_dtype)
        hidden_states = self.norm(norm_input).to(self.operator_dtype)
        if early_context is not None:
            return hidden_states, early_context
        if aux_hidden_states:""")
sources=dict(event.sources)
sources[MODEL]=source
def rep(a,b):
    assert sources[LP].count(a)==1,a
    sources[LP]=sources[LP].replace(a,b)
rep("    dflash_context_ready_event: Optional[torch.cuda.Event] = None","    dflash_context_ready_event: Optional[torch.cuda.Event] = None\n    dflash_tapped_context: Optional[torch.Tensor] = None")
rep('            dflash_context_ready_event=getattr(forward_batch, "dflash_context_ready_event", None),','''            dflash_context_ready_event=getattr(forward_batch, "dflash_context_ready_event", None),
            dflash_tapped_context=getattr(forward_batch, "dflash_tapped_context", None),''')
rep('                event.record()\n                context_ready = (event, hidden_states_to_store)','''                if hidden_states_to_store is not logits_metadata.dflash_tapped_context:
                    event.record()
                context_ready = (event, hidden_states_to_store)''')
for f,s in sources.items():
    ast.parse(s)
    original=subprocess.check_output(["git","-C",str(R/"sglang"),"show",event.base.PIN+":"+f]).decode()
    delta="diff --git a/"+f+" b/"+f+"\n"+"".join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),"a/"+f,"b/"+f))
    parts[f]=delta.encode()
    if f in (LP,MODEL):(D/("tap_"+Path(f).name)).write_text(s,encoding="utf8")
full=b"".join(parts.values());out=D/"tap_candidate.diff";out.write_bytes(full)
assert max(len(x[1:]) for x in full.splitlines() if x.startswith(b"+"))<=400
env={**os.environ,"GIT_INDEX_FILE":str(D/"tap.index")}
for args in (["read-tree",event.base.PIN],["apply","--cached","--check",str(out)]):
    subprocess.run(["git","-C",str(R/"sglang"),*args],env=env,check=True)
print("tap",len(full),hashlib.sha256(full).hexdigest())
