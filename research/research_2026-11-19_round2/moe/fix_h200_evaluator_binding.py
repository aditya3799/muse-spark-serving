from pathlib import Path
p=Path(__file__).with_name('evaluate_h200_receipt.py');s=p.read_text(encoding='utf-8-sig')
s=s.replace("def number(value):", "def canonical_hash(value):\n return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()\n\ndef number(value):")
needle=" require(sha(serving.get('integrated_patch_sha256')),'integrated_patch_binding')"
new=needle+'''
 manifest=serving.get('measured_source_manifest',{})
 require(manifest.get('patch_sha256')==serving.get('integrated_patch_sha256'),'measured_patch_binding')
 require(manifest.get('baseline_sha256')==PARENT,'measured_baseline_binding')
 require(manifest.get('kernel_sources')==SOURCES,'measured_kernel_sources')
 require(manifest.get('checkpoint_manifest_sha256')==receipt.get('checkpoint_manifest_sha256'),'measured_checkpoint_binding')
 require(manifest.get('route_holdout_sha256')==provenance.get('holdout_sha256'),'measured_route_binding')
 policy=serving.get('selected_policy',{})
 require(set(policy)=={'decode','prefill','fallback'},'selected_policy_scope')
 require(all(v in set(VARIANTS)|{'humming_parent'} for v in policy.values()),'selected_policy_variants')
 require(manifest.get('selected_policy')==policy,'measured_policy_binding')
 require(serving.get('measured_source_manifest_sha256')==canonical_hash(manifest),'source_manifest_hash')
'''
assert s.count(needle)==1;s=s.replace(needle,new)
s=s.replace("  require(sha(row.get('receipt_sha256')),prefix+'receipt')", "  require(sha(row.get('receipt_sha256')),prefix+'receipt')\n  require(row.get('source_manifest_sha256')==serving.get('measured_source_manifest_sha256'),prefix+'measured_source_binding')")
s=s.replace("conclusive_2x_evidence=not failures,scored_submission_authorized=False,", "reported_gates_consistent=not failures,conclusive_2x_evidence=False,\n             independent_frontier_recomputation_complete=False,scored_submission_authorized=False,")
s=s.replace("Receipt consistency only. Does not authenticate hardware or authorize a submission. No component rewards are summed or multiplied.","Reported-receipt consistency only. Official frontier scalars are not independently recomputed or authenticated here; conclusive_2x_evidence is always false. Human review must verify full raw official results. No component rewards are summed or multiplied.")
p.write_text(s)
p=p.with_name('test_h200_receipt.py');s=p.read_text(encoding='utf-8-sig')
needle=" return r\n\nbase=valid()"
new=''' policy=dict(decode='finite_only',prefill='humming_parent',fallback='unchanged_lut')
 manifest=dict(patch_sha256='4'*64,baseline_sha256=PARENT,kernel_sources=SOURCES,checkpoint_manifest_sha256='1'*64,route_holdout_sha256='3'*64,selected_policy=policy)
 r['serving'].update(measured_source_manifest=manifest,measured_source_manifest_sha256=canonical_hash(manifest),selected_policy=policy)
 for row in r['serving']['scenarios']:row['source_manifest_sha256']=canonical_hash(manifest)
 return r

base=valid()'''
assert s.count(needle)==1;s=s.replace(needle,new)
s=s.replace("assert result['decision']=='READY_FOR_HUMAN_REVIEW' and result['scored_submission_authorized'] is False", "assert result['decision']=='READY_FOR_HUMAN_REVIEW' and result['scored_submission_authorized'] is False\nassert result['reported_gates_consistent'] and not result['conclusive_2x_evidence']")
needle="reject('wrong_hardware'"
idx=s.index(needle)
s=s[:idx]+'''reject('unbound_integrated_patch',lambda r:r['serving'].update(integrated_patch_sha256='a'*64))
reject('unbound_selected_policy',lambda r:r['serving']['selected_policy'].update(decode='dependent_hint'))
reject('unbound_scenario_manifest',lambda r:r['serving']['scenarios'][0].update(source_manifest_sha256='b'*64))
reject('unbound_measured_checkpoint',lambda r:r['serving']['measured_source_manifest'].update(checkpoint_manifest_sha256='c'*64))
'''+s[idx:]
s=s.replace("out=dict(complete=True", "# Reported scalar claims alone can never establish independently conclusive evidence.\nr=valid()\nfor row in r['serving']['scenarios']:row['throughput_ratio']=.01\nassert evaluate(r)['conclusive_2x_evidence'] is False\npassed.append(dict(case='unverified_frontier_claim_never_conclusive',veto='independent frontier verification absent'))\nout=dict(complete=True")
p.write_text(s)
