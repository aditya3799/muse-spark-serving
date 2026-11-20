from pathlib import Path
p=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/moe')
for name in ['humming_probe.py','submission_humming_fp32_v1.py']:
 s=(p/name).read_text(encoding='utf-8-sig')
 s=s.replace("    return layer\n", "    import copy\n    layer.fp32_configs={}\n    for stage in ('w13','w2'):\n        default=HummingMethod.get_default_tuning_configs(layer,use_f16_accum=False,gemm_type='indexed',sublayer_name=stage)\n        snapshot=copy.deepcopy(default)\n        owned=copy.deepcopy(default)\n        for lo,hi,c in owned:\n            c['use_stream_k']=False\n            c['use_f16_accum']=False\n        assert default==snapshot, 'Default tuning cache must stay unchanged'\n        layer.fp32_configs[stage]=owned\n    return layer\n",1)
 start=s.index("    configs={s:HummingMethod.get_default_tuning_configs")
 end=s.index("    valid=local_ids.numel()",start)
 s=s[:start]+"    configs=layer.fp32_configs\n"+s[end:]
 (p/name).write_text(s)
print('owned tuning configs installed')
