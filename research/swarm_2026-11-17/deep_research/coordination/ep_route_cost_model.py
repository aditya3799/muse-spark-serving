import hashlib,json,math,pathlib,random
ROOT=pathlib.Path(r'D:\OpenKernelChallenge\serving')
OUT=ROOT/'swarm_2026-11-17/deep_research/coordination'
source=ROOT/'swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json'
meta=json.loads(source.read_text())
def config(stage,m):
    return next(c for lo,hi,c in meta['stages'][stage]['intervals'] if lo<m<=hi)
def route_counts(m,seed,pattern):
    rng=random.Random(seed)
    counts=[0]*256
    for row in range(m):
        if pattern=='uniform': ids=rng.sample(range(256),8)
        elif pattern=='rank_hot': ids=rng.sample(range(32),8)
        elif pattern=='expert_hot': ids=[0]+rng.sample(range(1,256),7)
        elif pattern=='half_rank_hot': ids=rng.sample(range(32 if row%2 else 256),8)
        elif pattern=='rank_balanced': ids=[32*r+rng.randrange(32) for r in range(8)]
        for e in ids: counts[e]+=1
    return [counts[i:i+32] for i in range(0,256,32)]
def cost(counts,policy,mode):
    vals=[]
    for rank in counts:
        padded={}; tiles={}
        for stage in ('w13','w2'):
            heavy,light=policy[stage]
            cutoff=min(policy[s][1]['block_shape'][0] for s in ('w13','w2'))
            rowtiles=0; paddedrows=0
            for count in rank:
                bm=heavy['block_shape'][0] if mode=='baseline' or (mode=='split' and count>cutoff) else light['block_shape'][0]
                nt=(count+bm-1)//bm
                rowtiles+=nt;paddedrows+=nt*bm
            n=meta['stages'][stage]['meta']['shape_n'];k=meta['stages'][stage]['meta']['shape_k']
            tiles[stage]=rowtiles
            padded[stage]={'rows':paddedrows,'weight_bytes_tile_model':rowtiles*n*k*(.5+1/16),'multiply_adds_padded':paddedrows*n*k}
        vals.append({'active_routes':sum(rank),'active_experts':sum(c>0 for c in rank),'stages':padded,'row_tiles':tiles})
    return vals
results=[]
for m in (1,8,16,32,64,128,256,512,768,1024,1536,3072,4096,8192,16384):
    policies={s:(config(s,m*8),config(s,m)) for s in ('w13','w2')}
    plans={s:{'baseline_block':p[0]['block_shape'],'light_block':p[1]['block_shape'],'baseline_warp':p[0]['warp_shape'],'light_warp':p[1]['warp_shape']} for s,p in policies.items()}
    for pattern in ('uniform','rank_balanced','expert_hot','rank_hot','half_rank_hot'):
        for seed in range(3):
            counts=route_counts(m,seed,pattern)
            costs={mode:cost(counts,policies,mode) for mode in ('baseline','split','single_light')}
            summary={}
            for mode,ranks in costs.items():
                summary[mode]={'max_rank_weight_bytes_tile_model':max(sum(t['weight_bytes_tile_model'] for t in r['stages'].values()) for r in ranks),'max_rank_padded_multiply_adds':max(sum(t['multiply_adds_padded'] for t in r['stages'].values()) for r in ranks),'max_rank_active_routes':max(r['active_routes'] for r in ranks),'min_rank_active_routes':min(r['active_routes'] for r in ranks)}
            results.append({'m':m,'pattern':pattern,'seed':seed,'plans':plans,'summary':summary})
report={'status':'STRUCTURAL_ROUTE_COST_MODEL_ONLY','source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Pinned source-mocked SM90 policies. GPU route counts simulated from unique top8 choices; no H200 timing, cache/bandwidth/collective model or serving forecast. K preservation does not affect this M-tile count; legality and numerical proof remain separate.','critical_path':'Compare the slowest rank, not sum of unrelated rank medians. An empty rank microregression need not worsen EP barrier latency.','limitations':['Policy modeled at shape rows, without whole-model graph padding/scheduling trace.','Weight bytes assume every row tile rereads full weight once; cache reuse and N tiling ignored.','Padded multiply-add count includes arithmetic potentially skipped by implementation; inspect actual scheduling before interpreting.','Single-light row policy may increase repeated weight reads for concentrated experts despite lower padding.'],'cases':results}
(OUT/'EP_ROUTE_COST_MODEL.json').write_text(json.dumps(report,indent=2)+'\n')
print('cases',len(results))
for row in results:
    if row['seed']==0 and row['m'] in (8,128,768,1024,3072) and row['pattern'] in ('uniform','rank_hot'):
        base=row['summary']['baseline'];split=row['summary']['split'];single=row['summary']['single_light']
        print(row['m'],row['pattern'],'split_weight_ratio',round(split['max_rank_weight_bytes_tile_model']/base['max_rank_weight_bytes_tile_model'],3),'split_padding_ratio',round(split['max_rank_padded_multiply_adds']/base['max_rank_padded_multiply_adds'],3),'single_weight_ratio',round(single['max_rank_weight_bytes_tile_model']/base['max_rank_weight_bytes_tile_model'],3))
