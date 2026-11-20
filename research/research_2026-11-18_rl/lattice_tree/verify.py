"""Check prefix choice against exhaustive enumeration and verify every committed token and KV slot against serial target attention."""
import itertools
import json
from pathlib import Path
import torch
from selector_tree import build_tree, walk

HERE=Path(__file__).resolve().parent


def audit_selection():
    cases = 0
    for depth,choices in ((2,3),(3,3),(4,2)):
        ids = torch.arange(depth*choices).view(1,depth,choices)
        for seed in (11,25,72):
            g=torch.Generator().manual_seed(seed)
            logits=torch.randn(1,depth,choices,choices,generator=g)
            coverage=.78+.21*torch.rand(1,depth,choices,generator=g)
            edge=(logits.float().log_softmax(-1)+coverage.log()[...,None])[0]
            best=[]
            for d in range(1,depth+1):
                for path in itertools.product(range(choices),repeat=d):
                    score=sum(float(edge[j,0 if j==0 else path[j-1],path[j]]) for j in range(d))
                    best.append((score,tuple(int(ids[0,j,t]) for j,t in enumerate(path))))
            for budget in (3,6,8):
                t,p,_,mask=build_tree(ids,logits,torch.tensor([9999]),budget,coverage)
                got=set()
                for j in range(1,budget):
                    chain=t[0][mask[0,j]].tolist()[1:]
                    assert int(p[0,j]) < j
                    got.add(tuple(chain))
                assert got == {tokens for _,tokens in sorted(best,reverse=True)[:budget-1]}
                cases+=1
    for nodes in (8,12,16):
        uniform=torch.zeros(2,7,16,16)
        ids=torch.arange(7*16).view(1,7,16).expand(2,-1,-1)
        target=build_tree(ids,uniform,torch.zeros(2,dtype=torch.long),nodes)
        assert (target[1][:,1:]>=0).all() and (target[3].sum(-1)==target[2]+1).all()
    return cases


def audit_attention(seed,nodes):
    g=torch.Generator().manual_seed(seed)
    v,c,k,d=43,16,16,7
    embed,key,value,head=[torch.randn(v,c,generator=g) for _ in range(4)]
    def predict(context):
        q=embed[context[-1]]; weight=(q@key[context].T/c**.5).softmax(-1)
        return int(((weight@value[context]+.3*q)@head.T).argmax())
    committed=holes=wrong_prefix=wrong_mask=0
    histories=[torch.randint(v,(9+r,),generator=g).tolist() for r in range(4)]
    for iteration in range(12):
        ids=torch.empty(4,d,k,dtype=torch.long)
        logits=torch.randn(4,d,k,k,generator=g)*.15
        for r,seq in enumerate(histories):
            temp=seq.copy()
            prev=0
            for j in range(d):
                token=predict(temp)
                options=torch.randperm(v,generator=g)
                options=options[options!=token][:k-1].tolist()
                target_rank=(j+r+iteration)%3
                options.insert(target_rank,token)
                ids[r,j]=torch.tensor(options)
                logits[r,j,:,0]+=2.5
                logits[r,j,prev,target_rank]+=2.6 if (j+r+iteration)%4 else 1.9
                prev=target_rank
                temp.append(token)
        t,p,position,mask=build_tree(ids,logits,torch.tensor([s[-1] for s in histories]),nodes)
        for r,seq in enumerate(histories):
            prefix=seq[:-1]
            all_tokens=prefix+t[r].tolist()
            queries=embed[t[r]]
            all_scores=queries@key[all_tokens].T/c**.5
            mask_full=torch.cat((torch.ones(nodes,len(prefix),dtype=torch.bool),mask[r]),dim=1)
            def predict_mask(attn_mask):
                weights=all_scores.masked_fill(~attn_mask,-float("inf")).softmax(-1)
                return ((weights@value[all_tokens]+.3*queries)@head.T).argmax(-1)
            target_next=predict_mask(mask_full)
            lower=torch.cat((torch.ones(nodes,len(prefix),dtype=torch.bool),torch.ones(nodes,nodes,dtype=torch.bool).tril()),dim=1)
            wrong_mask+=int((target_next!=predict_mask(lower)).sum())
            for j in range(nodes):
                assert int(target_next[j])==predict(prefix+t[r][mask[r,j]].tolist())
            sent,slots=walk(t[r],p[r],target_next,1+(iteration*3+r)%9)
            sliding=[all_tokens[i] for i in range(max(0,len(prefix)-5),len(prefix))]+[int(t[r,j]) for j in slots]
            expected=[]
            for _ in sent:
                expected.append(predict(seq+expected))
            assert sent==expected and sliding[-len(slots):]==[seq[-1],*sent[:-1]]
            wrong_prefix+=sum(int(t[r,i])!=int(t[r,j]) for i,j in enumerate(slots))
            holes+=sum(i!=j for i,j in enumerate(slots));committed+=len(sent)
            seq.extend(sent)
    assert holes and wrong_prefix and wrong_mask
    return dict(nodes=nodes,seed=seed,committed=committed,noncontiguous_slot_positions=holes,
                incorrect_values_with_naive_contiguous_cache=wrong_prefix,
                contaminated_predictions_with_flat_causal_mask=wrong_mask)


if __name__=="__main__":
    torch.set_num_threads(1)
    data=dict(scope="CPU exact-tree and causal target simulation. Independent serial attention supplies every accepted token. Neural logits are fixtures, so this is a correctness check only.",
              exhaustive_cases=audit_selection(),cases=[audit_attention(seed,n) for seed in (4409,5511) for n in (8,12,16)])
    (HERE/"verification.json").write_text(json.dumps(data,indent=2))
    print(data["exhaustive_cases"],sum(c["committed"] for c in data["cases"]))
