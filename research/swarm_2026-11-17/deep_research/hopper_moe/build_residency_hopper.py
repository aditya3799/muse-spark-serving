"""Prepare actual-Hopper exact residency gate; never executed on local SM86."""
from pathlib import Path
import hashlib
P=Path(__file__).resolve().parent
OLD=P.parents[2]/'swarm_2026-11-16/moe/phase4_hour'
source=OLD/'producer_fullchain_probe.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='a364344f79eb5ff8f8feb2511971cd46d7459e464dd9afe3be82e328d46863ea'
s=source.read_text()
s=s.replace("OWN=Path(__file__).resolve().parent;HERE=OWN.parent;ROOT=HERE.parents[1]", "OWN=Path(__file__).resolve().parent;ROOT=OWN.parents[2];HERE=ROOT/'swarm_2026-11-16/moe'")
s=s.replace("choices=['tma_b_only','tma_b'],default='tma_b_only'", "choices=['cp_async_s3_cta2','cp_async_s4_cta2','tma_b_only_s3_cta2','tma_b_only_s4_cta2','tma_b_s3_cta2','tma_b_s4_cta2'],default='cp_async_s4_cta2'")
s=s.replace('from producer_layer_view import producer_layer_view', 'from residency_view import residency_layer_view')
s=s.replace("candidate_layer,producer_manifest=producer_layer_view(humm,args.policy,90)", "producer,suffix=args.policy.rsplit('_s',1)\n    stages,ctas=(int(x) for x in suffix.split('_cta'))\n    candidate_layer,producer_manifest=residency_layer_view(humm,stages,ctas,90,producer)\n    producer_manifest['enabled']=bool(producer_manifest['changed'])")
s=s.replace("('producer_tuning.py','producer_layer_view.py')", "('residency_view.py',)")
s=s.replace("Hopper scored helper A versus producer-only TMA candidate B; same canonical tensors and GEMM geometry; runtime correctness and coldA/B gate", "Hopper scored helper A versus blockM8 residency candidate B including adaptive light arms M15-112; unchanged canonical tensors/block/warp/K/FP32. Exact runtime gates precede cold A/B.")
s=s.replace('relative<=.003 and maximum<=.03125', 'relative<=.003 and maximum<=.03125 and bool(torch.equal(value.view(torch.int16),reference.view(torch.int16)))')
s=s.replace('default=[33,128]', 'default=[1,8,14,15,32,112,113,128]')
s=s.replace("p.add_argument('--correctness-only'","p.add_argument('--reference-policy',choices=['scored_default','cp_async_s3_cta2','cp_async_s4_cta2'],default='scored_default');p.add_argument('--correctness-only'")
s=s.replace("producer_manifest['enabled']=bool(producer_manifest['changed'])","""producer_manifest['enabled']=bool(producer_manifest['changed'])
    reference_layer=humm
    reference_manifest={'policy':'scored_default'}
    if args.reference_policy!='scored_default':
        ref_stages=int(args.reference_policy.split('_s')[1].split('_')[0])
        reference_layer,reference_manifest=residency_layer_view(humm,ref_stages,2,90,'cp_async')
    producer_manifest['reference']=reference_manifest""")
s=s.replace("def scored():return finish(*helper_ns['_humming_expert_outputs'](humm,","def scored():return finish(*helper_ns['_humming_expert_outputs'](reference_layer,")
s=s.replace("raw_a,_=helper_ns['_humming_expert_outputs'](humm,","raw_a,_=helper_ns['_humming_expert_outputs'](reference_layer,")
(P/'residency_probe_sm90.py').write_text(s)
print('Hopper harness prepared:',hashlib.sha256(s.encode()).hexdigest())
