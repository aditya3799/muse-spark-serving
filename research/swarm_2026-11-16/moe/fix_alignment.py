from pathlib import Path
p=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/moe')
for name in ['humming_probe.py','submission_humming_fp32_v1.py']:
 s=(p/name).read_text(encoding='utf-8-sig')
 s=s.replace("    # Tuning both stages with one route table requires identical M blocks.\n    assert selected['w2']['block_shape'][0]==block,(selected['w13'],selected['w2'])\n", "")
 old="    HummingMethod.forward_layer(layer,inputs=cache2,outputs=cache3,top_k=1,sublayer_name='w2',tuning_config=configs['w2'],**common)"
 new="    if selected['w2']['block_shape'][0]!=block:\n        second=moe_align_block_size(local_ids,selected['w2']['block_shape'][0],layer.w2_post_scale.numel(),ignore_invalid_expert=True)\n        common.update(sorted_ids=second[0],expert_ids=second[1],num_tokens_padded=second[2])\n"+old
 assert old in s;s=s.replace(old,new)
 (p/name).write_text(s)
print('independent stage routing alignment supported')
