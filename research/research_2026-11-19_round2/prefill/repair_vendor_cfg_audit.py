"""Authorized repair of parent's four immutable-artifact offline CFG audits."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
AUDIT=ROOT/'research_2026-11-19_round2/coordination/wgmma_hazard_audit'
replacement=r'''native_link=mod._link
# Exact vendor block-terminal successors supplement the conservative upstream
# CALL target+fallthrough model. BRA.DIV has register-controlled fallthrough even
# without an @predicate; treating it as unconditional loses real execution paths.
vendor_edges={};cfg_offsets=[]
for node_name,node in nodes.items():
 targets=set(re.findall(re.escape('"'+node_name+'"')+r':[^\n]*? -> "([^"]+)"',cfg))
 assert all(t in nodes for t in targets),('Unknownvendorlabel',node_name,targets)
 address=node['last'];assert address not in vendor_edges,('Duplicateterminal',address)
 vendor_edges[address]={nodes[t]['first'] for t in targets}
 cfg_offsets.extend(int(x,16) for x in re.findall(r'([0-9a-f]{4,}):\\',node['body']))
assert len(cfg_offsets)==len(set(cfg_offsets)), 'Overlappingvendorblocks'
link_proof={}

def validate_vendor_edges(ins):
 idx={x.addr:i for i,x in enumerate(ins)}
 for address,expected in vendor_edges.items():
  assert address in idx
  actual={ins[i].addr for i in ins[idx[address]].succ}
  assert expected<=actual,('MissingvendorCFGedge',hex(address),expected-actual)

def resolved_link(ins):
 native_link(ins);idx={x.addr:i for i,x in enumerate(ins)}
 assert len(idx)==len(ins),'DuplicateSASSaddress'
 assert set(cfg_offsets)<=set(idx),'VendorinstructionmissingfromSASS'
 omitted=set(idx)-set(cfg_offsets)
 # nvdisasm omits alignment NOPs after the final RET. No executable instruction
 # or GMMA may disappear from the artifact-linked CFG.
 assert all(ins[idx[a]].op=='NOP' and a>max(cfg_offsets) for a in omitted),('UncoverednonpaddingSASS',omitted)
 observed=set();changes=[]
 for i,x in enumerate(ins):
  base=x.op.split('.')[0]
  if base in ('BRX','JMX'):
   assert base=='BRX' and x.addr in branches and not x.pred,('Unknownindirectbranch',x.text)
   m=re.fullmatch(r'BRX R8 -0x([0-9a-f]+)',x.text);assert m and int(m.group(1),16)==x.addr+16
   assert set(branches[x.addr])==vendor_edges[x.addr]
   observed.add(x.addr)
  elif base in ('BRA','JMP','CALL'):
   targets=[int(m.group(1),16) for o in x.ops if (m:=re.match(r'^`?\(?(0x[0-9a-f]+)',o))]
   assert len(targets)==1 and targets[0] in idx,('Unresolveddirectcontrol',x.text)
  if base in ('BRX','JMX','BRA','JMP','CALL','RET','EXIT','BPT','KILL'):
   assert x.addr in vendor_edges,('Controlnotavendorblockterminal',x.text)
  if x.addr in vendor_edges:
   before={ins[n].addr for n in x.succ};expected=vendor_edges[x.addr]
   assert expected<=set(idx),'VendoredgemissingfromSASS'
   after=expected if base=='BRX' else before|expected
   if before!=after:changes.append(dict(address=hex(x.addr),op=x.text,added=[hex(a) for a in sorted(after-before)]))
   x.succ=[idx[a] for a in sorted(after)];x.back=any(a<=x.addr for a in after)
 assert observed==set(branches)
 validate_vendor_edges(ins)
 # Call summaries are safe for these artifacts only because neither helper
 # creates nor retires pending GMMA state. All callee register writes remain
 # analyzed from the same incoming state via the retained CALL target edge.
 helpers=[]
 for call in (x for x in ins if x.op.split('.')[0]=='CALL'):
  target=next(int(m.group(1),16) for o in call.ops if (m:=re.match(r'^`?\(?(0x[0-9a-f]+)',o)))
  if any(h['entry']==hex(target) for h in helpers):continue
  todo=[idx[target]];seen=set();returns=[]
  while todo:
   i=todo.pop()
   if i in seen:continue
   seen.add(i);x=ins[i];base=x.op.split('.')[0]
   assert x.gmma is None and x.depbar is None,('CalleechangesGMMAstate',x.text)
   assert base not in ('CALL','BRX','JMX','EXIT'),('Unsupportedcalleecontrol',x.text)
   if base=='RET':returns.append(x.addr);continue
   todo.extend(x.succ)
  assert returns,'Calleewithoutreturn'
  helpers.append(dict(entry=hex(target),instructions=len(seen),returns=[hex(a) for a in sorted(returns)]))
 todo=[0];seen=set()
 while todo:
  i=todo.pop()
  if i in seen:continue
  seen.add(i);todo.extend(ins[i].succ)
 all_rs={i for i,x in enumerate(ins) if x.gmma and x.gmma['rs']}
 assert all_rs and all_rs<=seen,('UnreachableRSinventory',len(all_rs-seen))
 link_proof.update(vendor_blocks=len(vendor_edges),vendor_edges=sum(map(len,vendor_edges.values())),all_vendor_edges_covered=True,
  omitted_trailing_nops=len(omitted),reachable_rs=len(all_rs),repaired_successors=changes,call_helpers=helpers)
 return True
'''
for bundle in [AUDIT]+[AUDIT/n for n in ('wgs_register_0','wgs_split_ready_2','wgs_split_dependency_2')]:
 path=bundle/'audit_vendor_cfg.py';old=path.read_text()
 backup=bundle/'audit_vendor_cfg_before_bra_div_repair.py'
 if not backup.exists():backup.write_text(old)
 receipt=bundle/'finite_vendor_cfg_audit.json';saved=bundle/'finite_vendor_cfg_audit_before_bra_div_repair.json'
 if receipt.exists() and not saved.exists():saved.write_bytes(receipt.read_bytes())
 start=old.index('native_link=mod._link');end=old.index("report={'scope':",start)
 new=old[:start]+replacement+'\n'+old[end:]
 new=new.replace('only four BRX successors resolved from NVIDIA vendor CFG and exact cubin constant2 table.','all vendor block-terminal edges covered, four BRX targets matched to exact cubin constant2 table, and call helpers verified GMMA-state-neutral.')
 marker="report['artifact_binding_sha256']=hashlib.sha256((HERE/'vendor_output_binding.json').read_bytes()).hexdigest()"
 extra=r'''report['cfg_coverage']=dict(link_proof)
# Coverage negative control: dropping a known BRA.DIV fallthrough must reject,
# independently of whether a particular dataflow payload happens to expose it.
edge_neg=mod.parse_sass(sass)[next(iter(binding['expected_rs_inventory']))]
resolved_link(edge_neg)
div=next(x for x in edge_neg if x.op=='BRA.DIV')
next_addr=div.addr+16;idx_neg={x.addr:i for i,x in enumerate(edge_neg)}
assert next_addr in vendor_edges[div.addr]
div.succ.remove(idx_neg[next_addr])
try:
 validate_vendor_edges(edge_neg)
except AssertionError as exc:
 report['negative_omitted_bra_div_edge']={'address':hex(div.addr),'omitted':hex(next_addr),'detected':True,'error':str(exc)}
else:
 raise AssertionError('MissingBRA.DIVedgewasnotrejected')
'''
 new=new.replace(marker,extra+'\n'+marker)
 new += "\nreport['passed']=bool(report['complete'] and report['all_analyses_complete'] and report['problem_count']==0 and report['negative_controls_detected'] and report['negative_omitted_bra_div_edge']['detected'] and report['cfg_coverage']['all_vendor_edges_covered']);save()\nassert report['passed'],'StaticvendorCFGhazardgatefailed'\n"
 path.write_text(new)
print('Repaired four scripts; prior source/report snapshots preserved.')
