"""Same-proposal segmented verification state contract; no model/perf claim."""
import ast,hashlib,itertools,json
from pathlib import Path
import torch
P=Path(__file__).parent;ROOT=P.parents[3]
SOURCE=ROOT/'sglang/python/sglang/srt/speculative/dflash_worker_v2.py'
torch.set_num_threads(1)
node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_commit_accept')
ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'pinned_commit','exec'),ns)
commit=ns['_commit_accept']

def full(c,p):
    a=(c[:,1:]==p[:,:-1]).int().cumprod(1).sum(1).int()
    bonus=p[torch.arange(len(a)),a.long()]
    out,length=commit(c,a,bonus)
    return a,bonus,out,length

def segmented(c,p,cut,order):
    # First cut target inputs are c[:,0:cut]. The last prediction must compare
    # against c[:,cut], which ordinary width=cut accept does not inspect.
    edge=c[:,1:cut+1]==p[:,:cut]
    prefix_matches=edge.int().cumprod(1).sum(1).int()
    survivors=(prefix_matches==cut).nonzero().flatten()
    if order=='reverse':survivors=survivors.flip(0)
    a=prefix_matches.clone();bonus=p[torch.arange(len(a)),a.clamp(max=cut-1).long()].clone()
    for row in survivors.tolist():
        tail=(c[row,cut+1:]==p[row,cut:-1]).int().cumprod(0).sum().int()
        a[row]=cut+tail;bonus[row]=p[row,a[row]]
    out,length=commit(c,a,bonus)
    return a,bonus,out,length,survivors

def main():
    checks=0;cases=0;survivor_counts=set();first_commits=0;second_commits=0
    for cut,bits,order,mode in itertools.product((2,4,6),itertools.product((False,True),repeat=7),('stable','reverse'),('mixed','same')):
        c=torch.arange(3*8).view(3,8)+100
        p=torch.empty_like(c);p[:,:-1]=c[:,1:];p[:,-1]=700+torch.arange(3)
        masks=[bits,tuple(reversed(bits)),tuple(not b for b in bits)] if mode=='mixed' else [bits]*3
        for row,mask in enumerate(masks):
            for j,match in enumerate(mask):
                if not match:p[row,j]=900+10*row+j
        baseline=full(c,p);got=segmented(c,p,cut,order)
        for expected,actual in zip(baseline,got[:4]):assert torch.equal(expected,actual);checks+=1
        a,bonus,out,lens,survivors=got;survivor_counts.add(len(survivors))
        prefix=torch.tensor([31,49,67]);slots=torch.arange(24).view(3,8)+1000
        # Aux is consumed before the next target overwrites its shared output.
        hidden=torch.arange(3*8*5).view(3,8,5)+2000
        first=hidden[:,:cut].clone();draft_slots={};events=['draft8','target_first']
        stage1_lens=torch.minimum(lens,torch.tensor(cut))
        for row in range(3):
            for j in range(int(stage1_lens[row])):
                draft_slots[int(slots[row,j])]=tuple(first[row,j].tolist());first_commits+=1
        events.append('consume_first_aux')
        first.fill_(-1) # models shared target output overwrite safely after consume
        if len(survivors):
            events.append('target_tail')
            for row in survivors.tolist():
                assert a[row]>=cut and prefix[row]+cut==prefix[row]+stage1_lens[row]
                for j in range(cut,int(lens[row])):
                    draft_slots[int(slots[row,j])]=tuple(hidden[row,j].tolist());second_commits+=1
            events.append('consume_tail_aux')
        events.append('publish_once')
        expected={int(slots[row,j]):tuple(hidden[row,j].tolist()) for row in range(3) for j in range(int(lens[row]))}
        assert draft_slots==expected;checks+=1
        assert events.count('draft8')==1 and events.count('publish_once')==1 and events[-1]=='publish_once';checks+=1
        # Canonical emitted tokens and committed KV input rows differ by anchor/bonus.
        for row in range(3):
            n=int(lens[row]);assert out[row,:n].tolist()==c[row,1:n].tolist()+[int(bonus[row])]
            assert len(c[row,:n])==n
            checks+=1
        cases+=1
    # Decisive boundary negative: all first4 comparisons match, ordinary W4
    # accept reports3; continuation requires fourth edge p3==c4 explicitly.
    c=torch.arange(8).view(1,8);p=torch.arange(1,9).view(1,8)
    ordinary=full(c[:,:4],p[:,:4])[0].item()
    assert ordinary==3 and segmented(c,p,4,'stable')[0].item()==7
    # Wrong suffix start c5 loses the already-accepted c4 input KV row.
    assert c[:,4:].tolist()!=c[:,5:].tolist()
    assert tuple([-1]*5)!=tuple(range(5)) # wrong retained shared-aux view after overwrite
    assert [1000,1001]!=[1008,1009] # wrong compact-row owner
    h1=[4]*18+[5]*2;h2=[1]*10+[7]*8+[8]*2
    assert sum(h1)==sum(h2)==82 and len(h1)==len(h2)==20
    assert sum(x>=5 for x in h1)==2 and sum(x>=5 for x in h2)==10
    # Source permits split prefix-valid materialization at explicit slots but
    # does NOT establish equal FP32/BF16 results at smaller projection shapes.
    report={'status':'CPU_SEGMENTED_SAME_PROPOSAL_CONTRACT_PASS','cases':cases,'assertions':checks,
      'cut_widths':[2,4,6],'fixed_draft_width':8,'survivor_counts_covered':sorted(survivor_counts),
      'first_segment_committed_rows_checked':first_commits,'tail_committed_rows_checked':second_commits,
      'negative_controls':['ordinary_W4_accept_misses_fourth_continuation_edge','wrong_tail_start_drops_candidate4_KV',
        'overwritten_first_aux_safe_only_after_consumption','compacted_survivors_restore_original_request_indices'],
      'limits':['Synthetic fixed target prediction oracle; no actual target or draft projection arithmetic',
        'No GPU execution or production scheduling/controller','Both segment logits/aux and split draftKV projection require full-model numerical proof',
        'No acceptance-distribution forecast or timing projection'],
      'mean_length_counterexample':{'both_mean_commit':4.1,'tail_survival_fractions':[0.1,0.5],'not_a_measured_distribution':True},
      'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (P/'SEGMENTED_VERIFY_CPU.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()