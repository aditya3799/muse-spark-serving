"""Compare the tree SWA suffix kernel with the pinned expanded page table on the FA3 read set."""
import argparse
import json
import os
from pathlib import Path
import torch
from tree_pages import fill_tree_suffix
here=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=p.parse_args()
    if args.device=="cpu":assert os.environ.get("TRITON_INTERPRET")=="1"
    dev=args.device
    cases=[];torch.set_num_threads(1)
    for nodes,lens0 in ((3,[3,511,8193]),(8,[1,2048,74160]),(16,[1030,8412])):
        batch=len(lens0);capacity=max(lens0)+31
        prefix=torch.empty((batch,capacity),device=dev,dtype=torch.int32)
        branches=torch.empty((batch*nodes,nodes),device=dev,dtype=torch.int32)
        dst=torch.full((batch*nodes,capacity+nodes),-1337,device=dev,dtype=torch.int32)
        counts=0
        for trial in range(3):
            g=torch.Generator(device=dev).manual_seed(1641+trial*51+nodes)
            lens=torch.tensor(lens0[trial%batch:]+lens0[:trial%batch],dtype=torch.int32,device=dev)
            prefix.random_(0,180000,generator=g)
            # Random target KV slot IDs at tree nodes, rendered into ancestor order as the pinned scatter does.
            pool=torch.randint(100,180000,(batch,nodes),device=dev,dtype=torch.int32,generator=g)
            parent=[0]+[int(torch.randint(j,(1,),generator=torch.Generator().manual_seed(773*j+trial))) for j in range(1,nodes)]
            lengths=[];branches.fill_(0)
            for r in range(batch):
                for j in range(nodes):
                    chain=[j]
                    while chain[-1]>0:chain.append(parent[chain[-1]])
                    chain.reverse()
                    branches[r*nodes+j,:len(chain)]=pool[r,chain]
                    lengths.append(len(chain))
            anc=torch.tensor(lengths,dtype=torch.int32,device=dev)
            mapping=torch.randperm(180000,generator=g,device=dev).long()
            fill_tree_suffix(dst,prefix,branches,lens,anc,mapping,2048,nodes)
            for r in range(batch):
                n=int(lens[r])
                for j in range(nodes):
                    rr=r*nodes+j;l=lengths[rr]
                    oracle=mapping[torch.cat((prefix[r,:n],branches[rr,:l])).long()].int()
                    for tile in (32,64,128,256):
                        begin=max(0,(n+l-1-2048)//tile)*tile
                        assert torch.equal(dst[rr,begin:n+l],oracle[begin:]),(rr,n,begin,tile)
                        counts+=n+l-begin
        cases.append(dict(nodes=nodes,request_lengths=lens0,physical_page_values_checked=counts,reuses=3))
    result=dict(device=dev,interpreter=args.device=="cpu",map_type="int64",cases=cases,
                scope="Tree-expanded SWA metadata only. The dense reference concatenates the pinned prefix and sorted ancestors for each query. Every page ID in four conservative tile-aligned read sets is compared; no FA3 core time is measured.")
    (here/("tree_pages_"+dev+".json")).write_text(json.dumps(result,indent=2))
    print([c["physical_page_values_checked"] for c in cases],flush=True)
if __name__=="__main__":main()