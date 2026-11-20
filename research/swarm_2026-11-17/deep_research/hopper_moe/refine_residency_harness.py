from pathlib import Path
P=Path(__file__).resolve().parent
p=P/'residency_view.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("old=deepcopy(c)","""if sm_version==90:
                expected_geometry={'w13':([8,256,128],[8,32,64]),'w2':([8,128,256],[8,32,64])}
                if stage not in expected_geometry or (c['block_shape'],c['warp_shape'])!=expected_geometry[stage]:
                    raise ValueError('Uncompiled Hopper geometry; leave serving fallback untouched')
                if c.get('num_sms',0)<0:
                    raise ValueError('Invalid SM launch count')
            old=deepcopy(c)""")
p.write_text(s)
p=P/'build_residency_hopper.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("(P/'residency_probe_sm90.py').write_text(s)","""s=s.replace('default=[33,128]', 'default=[1,8,14,15,128]')
s=s.replace(\"p.add_argument('--correctness-only'\",\"p.add_argument('--reference-policy',choices=['scored_default','cp_async_s3_cta2','cp_async_s4_cta2'],default='scored_default');p.add_argument('--correctness-only'\")
s=s.replace(\"producer_manifest['enabled']=bool(producer_manifest['changed'])\",\"\"\"producer_manifest['enabled']=bool(producer_manifest['changed'])
    reference_layer=humm
    reference_manifest={'policy':'scored_default'}
    if args.reference_policy!='scored_default':
        ref_stages=int(args.reference_policy.split('_s')[1].split('_')[0])
        reference_layer,reference_manifest=residency_layer_view(humm,ref_stages,2,90,'cp_async')
    producer_manifest['reference']=reference_manifest\"\"\")
s=s.replace(\"def scored():return finish(*helper_ns['_humming_expert_outputs'](humm,\",\"def scored():return finish(*helper_ns['_humming_expert_outputs'](reference_layer,\")
s=s.replace(\"raw_a,_=helper_ns['_humming_expert_outputs'](humm,\",\"raw_a,_=helper_ns['_humming_expert_outputs'](reference_layer,\")
(P/'residency_probe_sm90.py').write_text(s)""")
p.write_text(s)
