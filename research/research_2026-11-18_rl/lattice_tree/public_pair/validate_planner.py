"""Select the native planner pair only after checking random BF16 chains and tree semantics."""
import torch
from cuda_planner import CompiledPlanner
from selector_tree import build_tree

def prefixes(t,p):
    paths=[()]
    for i in range(1,len(t)):
        paths.append(paths[int(p[i])]+(int(t[i]),))
    return set(paths)

def validated():
    tree,chain = CompiledPlanner(),CompiledPlanner("chain")
    g = torch.Generator(device="cuda").manual_seed(917731)
    seen = 0
    for i in range(28):
        ids=torch.arange(112,device="cuda",dtype=torch.int64).view(1,7,16)
        base=torch.randn((1,7,16,16),device="cuda",generator=g)*([1.8,4.,9.][i%3])
        scores=base.bfloat16()
        anchor=torch.tensor([455],device="cuda")
        mass=-torch.rand((1,7),device="cuda",generator=g)
        temp=(0.65,1.,1.5)[i%3]
        fork=(1.,0.5)[i%2]
        covered=(i%2)==0
        tid,tp,depth,mask=tree.single(ids,scores,anchor,f"n8_t{temp}_m{int(covered)}_f{fork}",mass)
        reference=build_tree(ids,scores.float()/temp,anchor,8,
                             mass.exp()[:,:,None].expand(-1,-1,16) if covered else None,after_fork=fork)
        assert prefixes(tid.tolist(),tp)==prefixes(reference[0][0].tolist(),reference[1][0].tolist())
        assert (depth==mask.sum(-1)-1).all()
        for j in range(8):
            path=set()
            k=j
            while k not in path:
                path.add(k)
                k=tp[k]
            assert path==set(torch.nonzero(mask[j]).flatten().tolist())
        line=chain.chain(ids,scores,anchor)[0].tolist()
        outcomes=[455];prev=0
        for j in range(7):
            prev=int(scores[0,j,prev].argmax())
            outcomes.append(int(ids[0,j,prev]))
        assert line==outcomes
        seen+=1
    return tree,chain,seen
