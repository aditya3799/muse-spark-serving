"""Validate every tensor name and shape, plus both forward paths, before fetching large public weights."""
import json
import torch
from observe import HERE, Qwen3Config,Qwen3ForCausalLM,Qwen3RotaryEmbedding, Drafter, Tokenizer, fetch,TARGET
table=json.loads((HERE/"headers.json").read_text())
cfg=Qwen3Config(**json.loads((HERE/"target_config.json").read_text()))
cfg._attn_implementation="sdpa"
def mock(meta):
    return {k:torch.empty(v["shape"],dtype=torch.bfloat16,device="meta") for k,v in meta.items() if k!="__metadata__"}
with torch.device("meta"):
    model=Qwen3ForCausalLM(cfg)
target={}
for header in table["target"].values():target.update(mock(header))
model.load_state_dict(target,strict=True,assign=True)
small=mock(table["draft"]["model.safetensors"])
d=Drafter(small,target["model.embed_tokens.weight"],target["lm_head.weight"])
taps=torch.empty(14,8192,dtype=torch.bfloat16,device="meta")
ids=torch.empty(16,dtype=torch.long,device="meta")
chosen,scores,mass=d.block_logits(taps,ids,[9,10,12,14])
assert list(chosen.shape)==[4,7,16] and list(scores.shape)==[4,7,16,16] and list(mass.shape)==[4,7]
tokenizer=Tokenizer.from_str(fetch(TARGET+"tokenizer.json").decode())
assert tokenizer.token_to_id("<|im_start|>")==151644
print("Strict target state keys",len(target),"draft tensors",len(small),"lattice",scores.shape)
