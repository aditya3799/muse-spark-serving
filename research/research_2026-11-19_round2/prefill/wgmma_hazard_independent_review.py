"""Read-only independent artifact/control-flow review of parent's WGMMA checker."""
import hashlib,json,re,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
AUDIT=ROOT/'research_2026-11-19_round2/coordination/wgmma_hazard_audit'
CHECKER=AUDIT/'python__sglang__kernels__jit__utils__wgmma_rs_hazard.py'
spec=importlib.util.spec_from_file_location('hazard_review',CHECKER);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
variants=['wgs_finite_2','wgs_register_0','wgs_split_ready_2','wgs_split_dependency_2'];records=[]
for name in variants:
 bundle=AUDIT if name=='wgs_finite_2' else AUDIT/name
 binding=json.loads((bundle/'vendor_output_binding.json').read_text())
 cubin=ROOT/'research_2026-11-19_round2/moe'/f'{name}_sm90a.cubin'
 sasspath=cubin.with_name(f'{name}_sass.txt');sass=sasspath.read_text();cfg=(bundle/'finite_vendor_bbcfg.dot').read_text()
 for observed,key in [(hashlib.sha256(cubin.read_bytes()).hexdigest(),'cubin_sha256'),(hashlib.sha256(sass.encode()).hexdigest(),'sass_normalized_text_sha256'),(hashlib.sha256(cfg.encode()).hexdigest(),'cfg_sha256')]:assert observed==binding[key],(name,key)
 table=json.loads((bundle/'finite_constant_tables.json').read_text());blob=cubin.read_bytes()
 for s in table['sections']:
  actual=blob[s['offset']:s['offset']+s['size']]
  assert hashlib.sha256(actual).hexdigest()==s['sha256']
  assert s['size']%4==0
  assert [int.from_bytes(actual[i:i+4],'little') for i in range(0,len(actual),4)]==[int(x,16) for x in s['words']]
 fns=mod.parse_sass(sass);observed={fn:sum(bool(i.gmma and i.gmma['rs']) for i in ins) for fn,ins in fns.items() if any(i.gmma for i in ins)}
 assert observed==binding['expected_rs_inventory'] and sum(observed.values())==512
 f=next(iter(observed));ins=fns[f];idx={i.addr:n for n,i in enumerate(ins)};assert len(idx)==len(ins)
 mod._link(ins)
 calls=[i for i in ins if i.op.split('.')[0]=='CALL'];targets={}
 for c in calls:
  m=re.search(r'0x[0-9a-f]+',c.text);assert m
  target=int(m.group(),16);targets[target]=targets.get(target,0)+1
 helpers=[]
 for target,call_count in sorted(targets.items()):
  todo=[idx[target]];seen=set();rets=[]
  while todo:
   n=todo.pop()
   if n in seen:continue
   seen.add(n);i=ins[n];base=i.op.split('.')[0]
   assert base not in ('BRX','JMX','CALL','EXIT'),(name,hex(i.addr),i.text)
   if base=='RET':rets.append(i.addr);continue
   todo.extend(i.succ)
  assert rets
  gmma=[hex(ins[n].addr) for n in seen if ins[n].gmma]
  depbar=[hex(ins[n].addr) for n in seen if ins[n].depbar is not None]
  assert not gmma and not depbar,(name,target,gmma,depbar)
  helpers.append(dict(entry=hex(target),call_count=call_count,reachable_instructions=len(seen),returns=[hex(x) for x in sorted(rets)],gmma=0,depbar=0))
 report=json.loads((bundle/'finite_vendor_cfg_audit.json').read_text())
 assert report['complete'] and report['all_analyses_complete'] and report['problem_count']==0 and report['negative_controls_detected']
 records.append(dict(variant=name,hash_binding=True,constant_table_bytes_reparsed=True,rs_inventory=observed,calls=len(calls),call_helpers=helpers,reported_static_gate=True))
doc=dict(passed=True,scope=__doc__,checker_sha256=hashlib.sha256(CHECKER.read_bytes()).hexdigest(),variants=records,
 findings=['FourBRX and512RS are exact-artifact constraints, not workload dispatch conditions.','Existing helpers contain neither GMMA nor DEPbar, so upstreamcall+fallthrough overapproximation carries the same pending-groupstate afterreturn for THESE artifacts.','Generically a callee issuingGMMA would require return-state modeling; this review is not universal CFG validation.','Older audit_one.py accepts parsedfunctions with noGMMA; its complete flag alone is NOT a passing predicate.','New audit_vendor_cfg.py must be consumed using complete AND all_analyses_complete AND problem_count==0 AND negative_controls_detected, not process exit alone.'],
 limitations=['No GPUexecution, racecheck, timing, barrier/TMA correctness or model-output proof.','Trusts upstreamSASS instruction parser/state abstraction; exact binding does not make that analysis universal.'])
(HERE/'wgmma_hazard_independent_review.json').write_text(json.dumps(doc,indent=2));print(json.dumps(dict(passed=True,variants=len(records),calls_per_variant=[x['calls'] for x in records])))
