import hashlib,json,pathlib,datetime
here=pathlib.Path(__file__).resolve().parent;root=here.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review=root/'prefill/wgmma_bra_div_repair_review.json';summary=json.loads(review.read_text())
rows=[]
for row in summary['rows']:
 path=root/row['path'];report=json.loads(path.read_text())
 assert sha(path)==row['report_sha256'] and report['passed']
 assert report['cfg_coverage']['all_vendor_instructions_reachable']
 assert report['cfg_coverage']['all_vendor_edges_covered']
 rows.append(dict(path=row['path'],sha256=sha(path),rs=512,scope='static RS/accumulator only'))
for name in ['pr50800_offline','pr50800_boundaries']:
 path=root/'coordination/wgmma_hazard_audit'/name/'complete_cfg_audit.json';report=json.loads(path.read_text())
 assert report['static_screen_pass'] and report['problem_count']==0
 assert report['raw_vendor_edge_count']==report['parsed_vendor_edge_count']==1993
 assert all(all(x.values()) for x in report['coverage_negative_controls'].values())
 assert sha(root/'coordination/wgmma_hazard_audit/audit_donor_cfg.py')==report['adapter_sha256']
 rows.append(dict(path=str(path.relative_to(root)),sha256=sha(path),rs=384,scope='static RS/accumulator only'))
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prefill_summary_sha256=sha(review),
 independently_reviewed_successor_logic=True,repaired_reports=rows,hardware_execution=False,
 numerical_correctness_proven=False,async_memory_safety_proven=False,performance_proven=False,
 serving_ready=False,prior_failures_preserved=True)
(here/'final_static_cross_review.json').write_text(json.dumps(out,indent=2)+'\n');print('Six static receipts bound; no hardware/serving claim')
