"""Pack high-probability prefixes of the existing DFlash2 transition lattice into a fixed-size verification tree."""
import math
import torch


def build_tree(candidates, scores, bonus, nodes, coverage=None, after_fork=1.):
    """Inputs follow the pinned selector (B, depth, K, K). Node zero is the target bonus.
    Optional coverage is the retained mass per predecessor after truncation to K candidates.
    """
    batch, depth, choices = candidates.shape
    if scores.shape != (batch, depth, choices, choices) or not 2 <= nodes <= 1 + sum(choices**d for d in range(1, depth+1)):
        raise ValueError("invalid lattice dimensions or verification budget")
    branch = torch.log_softmax(scores.float(), dim=-1)
    if coverage is not None:
        if coverage.shape != scores.shape[:-1]:
            raise ValueError("invalid retained-mass shape")
        branch = branch + coverage.float().clamp(min=1e-30,max=1).log().unsqueeze(-1)
    if not 0 < after_fork <= 1:
        raise ValueError("after-fork coverage factor must lie in (0, 1]")
    width = nodes - 1
    device = scores.device
    row = torch.arange(batch,device=device)[:,None]
    previous_score = scores.new_zeros((batch,1),dtype=torch.float32)
    previous_choice = torch.zeros((batch,1),device=device,dtype=torch.long)
    forked = torch.zeros_like(previous_choice,dtype=torch.bool) if after_fork < 1 else None
    blocks = []
    previous_base = 0
    next_base = 1
    for d in range(depth):
        edge = branch[row,d,previous_choice]
        if forked is not None:
            edge = edge + forked[:,:,None]*math.log(after_fork)
        path_score = (previous_score[:,:,None] + edge).flatten(1)
        size = min(width,path_score.shape[1])
        # Stable ties retain the previous beam's deterministic order, then token index.
        take = path_score.argsort(dim=1,descending=True,stable=True)[:,:size]
        previous_score = path_score.gather(1,take)
        previous_choice = take.remainder(choices)
        if forked is not None:
            from_beam = take.div(choices,rounding_mode="floor")
            likely = edge.argmax(-1).gather(1,from_beam)
            forked = forked.gather(1,from_beam) | (likely != previous_choice)
        parent = take.div(choices,rounding_mode="floor") + previous_base
        tokens = candidates[:,d].gather(1,previous_choice)
        blocks.append((previous_score,parent,tokens))
        previous_base,next_base = next_base,next_base+size

    all_scores = torch.cat([block[0] for block in blocks],dim=1)
    # When the path probability is tied with its parent, the earlier (shallower) block wins.
    kept = all_scores.argsort(dim=1,descending=True,stable=True)[:,:width].sort(dim=1).values
    mapping = torch.full((batch,next_base),-1,device=device,dtype=torch.long)
    mapping[:,0] = 0
    local = torch.arange(1,nodes,device=device).expand(batch,-1)
    mapping.scatter_(1,kept+1,local)
    parents = torch.cat([block[1] for block in blocks],dim=1).gather(1,kept)
    parents = mapping.gather(1,parents)
    parents = torch.cat([torch.zeros_like(parents[:,:1]),parents],dim=1)
    tokens = torch.cat([bonus.view(batch,1),torch.cat([block[2] for block in blocks],dim=1).gather(1,kept)],dim=1)
    # A node can see precisely its ancestors and itself. The target prepends the full common prefix.
    self_ids = torch.arange(nodes,device=device).expand(batch,-1)
    mask = torch.eye(nodes,device=device,dtype=torch.bool).expand(batch,-1,-1).clone()
    ancestor = self_ids
    for _ in range(depth):
        ancestor = parents.gather(1,ancestor)
        mask.scatter_(2,ancestor[:,:,None],True)
    positions = mask.sum(-1)-1
    return tokens, parents, positions, mask


def walk(tokens, parents, target_next, max_emit):
    """Reference commit returns emitted tokens and accepted target KV node IDs, including the anchor."""
    if max_emit < 0:
        raise ValueError("negative remaining token budget")
    if max_emit == 0:
        return [],[]
    emitted = []
    accepted = [0]
    current = 0
    while len(emitted) < max_emit:
        target = int(target_next[current])
        emitted.append(target)
        if len(emitted) == max_emit:
            break
        children = [j for j in range(1,len(tokens)) if int(parents[j]) == current and int(tokens[j]) == target]
        if not children:
            break
        if len(children) != 1:
            raise ValueError("ambiguous sibling token")
        current = children[0]
        accepted.append(current)
    return emitted,accepted