from pathlib import Path
import hashlib
import json
import math
import numpy as np

root=Path(r'D:\OpenKernelChallenge\serving')
out=root/'research_1p3x_2026-11-13'
out.mkdir(exist_ok=True)
candidate=root/'submission_v4.diff'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == '92054389830a8e5c07a37eb524810ac83690fbbcf88c60e3ae8a132909bb65f1'

def tile(m, experts):
    for b in [8,16,32,48,64]:
        if m*8/experts/b < .9:
            return b
    return b

tile_rows=[]
simulation=[]
for users in [1,4,16,32]:
    m=users*8
    old=tile(m,32)
    density=tile(m,256)
    tile_rows.append(dict(users=users,verify_tokens=m,current_tile=old,density_tile=density,
        current_estimated_rows_per_local_expert=m*8/32,
        uniform_expected_rows_per_local_expert=m*8/256))
    for distribution in ['uniform','lognormal_sigma1']:
        observations=[]
        for seed in range(100):
            rng=np.random.default_rng(seed)
            logits=np.zeros(256) if distribution=='uniform' else rng.normal(size=256)
            ids=np.argpartition(logits[None,:]+rng.gumbel(size=(m,256)),-8,axis=1)[:,-8:]
            counts=np.bincount(ids.flatten(),minlength=256).reshape(8,32)
            counts=counts[counts.sum(1).argmax()]
            blocks_old=((counts+old-1)//old).sum()
            blocks_new=((counts+density-1)//density).sum()
            observations.append([counts.sum(),blocks_old*old,blocks_new*density,blocks_old,blocks_new])
        avg=np.mean(observations,axis=0)
        simulation.append(dict(users=users,distribution=distribution,mean_local_routes=avg[0],
            old_padded_rows=avg[1],density_padded_rows=avg[2],
            old_weight_tiles=avg[3],density_weight_tiles=avg[4],
            padded_row_ratio=avg[1]/avg[2],weight_tile_ratio=avg[4]/avg[3]))

stock=[150.56,206.62,462.51,620.51]
own=[67.04,122.86,174.47,208.04]
donor_speed=[s/c for s,c in zip(stock,own)]
scenarios=[]
for share in [.2,.3,.4,.5,.6]:
    rates=[1/(1-share+share/r) for r in donor_speed]
    scenarios.append(dict(assumed_moe_fraction_each_load=share,per_load_speedup=rates,
        overall_geomean=math.prod(rates)**.25))
memory_layer=32*3*4096*4096*(.5+1/16)
emax=384*8
fixf=2*160*4*4096+2*160*4+136*16
xrows=emax+8*(32+emax//32+8)
scratch=(8+1024+emax)*4+(emax+emax*4096+fixf)*4+(emax+2*xrows)*4096*2
paths={
    'own_moe_v139':root.parent/'muse_moe_w4a16/submission_v139_wgs_kperm3.py',
    'own_bf16_v150':root.parent/'muse_bf16_proj/submission_rs128_faithful_v150.py',
    'own_attention':root.parent/'muse_attention_decode/submission.py',
    'reference_moe_i4':root/'references/gauravjain14-muse-okc-submissions/muse_moe_w4a16/11_cold_start_end_chain_item_loop/build.py',
}
donors={k:dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()}
report=dict(target='1.3x geometric-mean serving score over 1,4,16,32 users',
    required_uniform_time_reduction=1-1/1.3,
    gain_if_only_one_load_is_1p3x=1.3**.25,
    one_load_gain_needed_if_other_three_are_1x=1.3**4,
    tile_heuristic=tile_rows,
    synthetic_padding_analysis=simulation,
    synthetic_analysis_limit='Structural simulation on generic weighted random routing, busiest EP rank; not GPU timing or a serving workload profile. More weight tiles can erase padding benefits.',
    own_moe_standalone_speedups_verify=donor_speed,
    minimum_moe_share_for_1p3x_if_donor_speed_transfers=[(1-1/1.3)/(1-1/r) for r in donor_speed],
    amdahl_scenarios=scenarios,
    amdahl_limit='Hypothetical equal MoE fraction per load; standalone kernel ratios do not establish serving speedup or transfer across baseline versions.',
    memory=dict(extra_full_weight_layout_bytes_per_layer=int(memory_layer),
        extra_full_layout_GiB_at_60_layers=memory_layer*60/2**30,
        extra_full_layout_GiB_at_62_layers=memory_layer*62/2**30,
        own_scratch_bytes_per_layout=scratch,
        own_two_scratch_layouts_GiB_at_60_layers=2*scratch*60/2**30,
        layer_count_note='60 is standalone-task weighting; serving config default is 62 layers. Count actual sparse layers before allocating.'),
    donors=donors,
    queued_candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
(out/'analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['tile_heuristic','synthetic_padding_analysis','minimum_moe_share_for_1p3x_if_donor_speed_transfers','amdahl_scenarios','memory','donors']},indent=2))
