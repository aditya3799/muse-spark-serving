import json,pathlib
p=pathlib.Path(__file__).resolve().parent
r=json.loads((p/'SM90_OCCUPANCY_REVIEW.json').read_text());occ=json.loads((p/'TIMING_OCCUPANCY_REVIEW.json').read_text())['occupancy']
for row in r['rows']:
 counts=next(v['counts'] for v in occ if v['m']==row['m'] and v['pattern']==row['pattern']);cut=row['cutoff']
 for stage,v in row['stages'].items():
  lm,hm=v['light_m'],v['heavy_m'];base=sum((n+hm-1)//hm for n in counts);candidate=sum((n+(lm if n<=cut else hm)-1)//(lm if n<=cut else hm) for n in counts)
  cfg=row['policy'][stage]
  n=8192 if stage=='w13' else 4096;k=4096
  v.update(light_NK=cfg['light']['block_shape'][1:],heavy_NK=cfg['heavy']['block_shape'][1:],baseline_N_tiles_per_Mtile=n//cfg['heavy']['block_shape'][1],light_N_tiles_per_Mtile=n//cfg['light']['block_shape'][1],baseline_M_tiles=base,candidate_M_tiles=candidate,logical_weight_scale_Mtile_ratio=base/candidate if candidate else None,logical_weight_plus_group_scale_bytes_per_Mtile=n*k//2+n*k//16,logical_requested_weight_plus_scale_bytes_baseline=base*(n*k//2+n*k//16),logical_requested_weight_plus_scale_bytes_candidate=candidate*(n*k//2+n*k//16))
r['limits'] += ['At cutoff=min(lightBM), every light expert occupies one M tile in both policies; unchanged heavy policy means M tile count is conserved. W13 has equal N/K tiles. W2 light BN128 versus heavy BN256 doubles N tile count while halving weight tile width; the full logical weight traversal bytes remain equal for these divisible dimensions.','Logical byte model assumes one full FP4+FP8-group-scale weight traversal per expert M tile. It is not measured HBM traffic and omits cache, multicast, physical load protocol and padding metadata.','Padded row ratios concern arithmetic/output work, not weight traffic. W2 light has twice as many N tiles and potentially repeats activation reads more often. WGMMA operand layout, dequantization and occupancy may reduce or remove expected transfer of SM86 speedup.']
(p/'SM90_OCCUPANCY_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps([{k:row[k] for k in ('m','pattern')} for row in r['rows']]))


