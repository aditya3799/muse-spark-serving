"""Inspect the first changed target decision, outside the measured interval, without resuming a divergent history."""
import torch

def collect(prefix,nodes,retained,index,logits,gold_token):
    row=int(retained[index])
    columns=torch.topk(logits[0,row],5)
    return dict(history=prefix.tolist()+nodes[retained[:index+1]].tolist(),branch_row=row,
        gold_token=int(gold_token),tree_token=int(columns.indices[0]),branch_top5=columns.indices.cpu().tolist(),
        branch_values=columns.values.float().cpu().tolist(),gold_branch_logit=float(logits[0,row,gold_token]))

def replay(model,entry):
    seq=torch.tensor(entry.pop("history"),device="cuda")[None,:]
    # All preceding target-cache entries in this fresh prefix are computed from the true path.
    logits=model(input_ids=seq,use_cache=False,logits_to_keep=1).logits[0,-1]
    val,ids=logits.topk(5)
    entry.update(prefix_tokens=seq.shape[1],fresh_top5=ids.tolist(),fresh_values=val.float().tolist(),
        fresh_gold=float(logits[entry["gold_token"]]),fresh_tree=float(logits[entry["tree_token"]]))
    return entry
