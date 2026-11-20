"""Bind the unchanged graph fixture to separate W13 and W2 native group-scale dtypes."""
from pathlib import Path
here=Path(__file__).resolve().parent
src=(here.parent/"narrow_n/paired.py").read_text()
tgt=(here.parent/"grouped_route/paired_fused.py").read_text()
helper=tgt[tgt.index("def finite_scales"):tgt.index("weights=weights.replace")]
pos=src.index("raw=torch.nn.Module();")
src=src[:pos]+helper+src[pos:]
src=src.replace('torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)','finite_scales(e,n,h)')
src=src.replace("from policy import narrow","from policy import expanded\nexpanded_layers={stage:expanded(layer,stage) for stage in ('w13','w2')}\nfrom humming.config import ComputeConfig,TuningConfig\nfrom humming.utils.device import fits_device_smem\nfrom dataclasses import fields")
src=src.replace('candidate="Full-K CTA-N 128"','candidate="Canonical BF16 per-16 group scales"')
start=src.index('    conf=narrow(old,')
end=src.index('    meta=layer.humming_metas',start)
src=src[:start]+'''    if capture:captured.append(dict(kw))
    conf=old
    if role in (stage,"both"):
        layer=expanded_layers[stage]
        meta=layer.humming_metas[stage]
        comp=ComputeConfig(**kw["compute_config"])
        keys={f.name for f in fields(TuningConfig) if f.init}
        config=TuningConfig(**{k:v for k,v in old.items() if k in keys})
        if not fits_device_smem(meta,comp,config,kw["inputs"].device):
            conf=dict(old,num_stages=2)
            config=TuningConfig(**{k:v for k,v in conf.items() if k in keys})
            assert fits_device_smem(meta,comp,config,kw["inputs"].device),(stage,old)
    kw["tuning_config"]=conf
'''+src[end:]
src=src.replace('role=role,stage=stage,id=kid,block=kernel.block_shape,warp=kernel.warp_shape,','role=role,stage=stage,id=kid,block=kernel.block_shape,warp=kernel.warp_shape,bs_dtype=str(meta.bs_dtype),stages=kernel.num_stages,')
src=src.replace('for i in range(8):','for i in range(8):')
src=src.replace('    env=DenseEnvelope(m,args.seed)','    print("start case",m,flush=True)\n    env=DenseEnvelope(m,args.seed)')
src=src.replace('    for name in ["parent"]+variants:\n        role=name;run();torch.cuda.synchronize()', '    for name in ["parent"]+variants:\n        print("capturing",m,name,flush=True)\n        role=name;run();torch.cuda.synchronize()')
src=src.replace('expanded_layers={stage:expanded(layer,stage) for stage in (\'w13\',\'w2\')}', 'expanded_layers={stage:expanded(layer,stage) for stage in ("w13","w2") if stage in args.variants or "both" in args.variants}')
src=src.replace('        if valid==0:continue', '        if valid==0 or (stage not in variants and "both" not in variants):continue')
(here/"paired.py").write_text(src)
score=(here.parent/"narrow_n/score.py").read_text().replace('"fixed_warp_narrow_cta_N_"','"native_bf16_group_scale_"')
(here/"score.py").write_text(score)
