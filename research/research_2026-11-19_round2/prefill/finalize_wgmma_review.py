from pathlib import Path
import json,hashlib,datetime
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'coordination/wgmma_hazard_audit'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for name,rel in [('finite',''),('register','wgs_register_0'),('split_ready','wgs_split_ready_2'),('split_dependency','wgs_split_dependency_2')]:
 d=BASE/rel;p=d/'finite_vendor_cfg_audit.json';r=json.loads(p.read_text());c=r['cfg_coverage']
 assert r['passed'] and r['problem_count']==0 and c['reachable_rs']==512
 assert c['raw_dot_edges']==c['vendor_edges'] and c['all_vendor_instructions_reachable']
 assert len([x for x in c['repaired_successors'] if x['op'].startswith('BRA.DIV')])==3
 assert r['negative_omitted_bra_div_edge']['detected'] and r['negative_controls_detected']
 rows.append(dict(variant=name,path=str(p.relative_to(HERE.parent)),report_sha256=h(p),audit_sha256=h(d/'audit_vendor_cfg.py'),before_report_sha256=h(d/'finite_vendor_cfg_audit_before_bra_div_repair.json'),before_audit_sha256=h(d/'audit_vendor_cfg_before_bra_div_repair.py'),passed=True,coverage=c,negative_omitted_bra_div_edge=r['negative_omitted_bra_div_edge'],negative_injected_write=r['functions'][0]['negative_injected_write']))
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS: supersedes all four pre-repair static reports only',scope='Artifact-bound offline RS/accumulator hazard screen; not H200 execution, numerical correctness, performance, or serving readiness.',repair='Every vendor block-terminal successor is included. BRX uses exact vendor targets bound to cubin constant2 tables. Other terminals retain native successors union vendor successors, preserving CALL callee traversal. Two callees per artifact have no GMMA/DEPbar or nested/indirect control. All vendor instruction addresses and all 512 RS sites reachable; omitted self-BRA/NOP tail proven disconnected.',controls='Per artifact: injected A-register write detected; omitted BRA.DIV fallthrough rejected. Exact cubin, SASS and CFG hashes bound; expected nonempty inventory enforced; raw DOT edge count equals parsed count.',intermediate_failures=[dict(cause='Overstrict NOP-only tail assertion rejected actual unreachable self-BRA trap',resolution='Require exact self-BRA/NOP classes, after all CFG addresses, no incoming covered edge, no entry reachability'),dict(cause='Copied finite R8-only guard rejected register variant R2',resolution='Restored original variant R[0-9]+ guard; exact PC offset and artifact-bound target evidence retained')],rows=rows)
(HERE/'wgmma_bra_div_repair_review.json').write_text(json.dumps(out,indent=2)+'\n')
text='''Four WGS vendor-CFG hazard audits repaired and rerun, 2026-11-19.

Prior passes were incomplete: native BRA handling omitted three unpredicated BRA.DIV fallthroughs in each binary. Earlier report/script copies are preserved as *_before_bra_div_repair.*. Current finite_vendor_cfg_audit.json files supersede them.

All four final audits pass with zero reported RS/accumulator hazards, all 512 RS sites reachable per artifact, complete vendor instruction reachability, and complete DOT-edge parsing. Vendor block/edge counts: finite1114/1588; register1174/1648; split-ready1210/1748; split-dependency1274/1812. All three BRA.DIV edges and four indirect branch sites are included per binary. Negative omitted-edge and injected A-write controls both detect the fault.

CALL callee edges remain conservatively present even though vendor DOT shows only fallthrough. Their two reachable helper bodies have no GMMA/DEPbar; all helper ordinary register writes remain analyzed. This is an artifact-specific summary, not a general interprocedural checker proof. Omitted terminal self-BRA/NOP padding is after all CFG instructions and has no incoming covered edge or entry reachability.

No frozen candidate, cubin, SASS, vendor CFG, upstream state analysis, GPU workload, or serving setting was changed. No native GPU correctness or speed conclusion follows. Root separately owns original/boundary PR50800 donor audits.

See wgmma_bra_div_repair_review.json for exact source/report hashes, controls, coverage and preserved earlier receipts.
'''
(HERE/'WGMMA_BRA_DIV_REPAIR_REVIEW.txt').write_text(text)
print(json.dumps(dict(passed=len(rows),review_sha256=h(HERE/'wgmma_bra_div_repair_review.json'))))
