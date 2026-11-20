import pathlib,re,json,hashlib,resource,importlib.util,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('hazard',HERE/'python__sglang__kernels__jit__utils__wgmma_rs_hazard.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
cfg=(HERE/'finite_vendor_bbcfg.dot').read_text();sasspath=pathlib.Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/moe/wgs_split_dependency_2_sass.txt');sass=sasspath.read_text()
receipt=json.loads((HERE/'vendor_bbcfg_receipt.json').read_text());tables=json.loads((HERE/'finite_constant_tables.json').read_text())
assert receipt['exit_code']==0 and receipt['cubin_sha256']==tables['sha256']
source=pathlib.Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/moe/wgs_split_dependency_2_sm90a.cubin')
assert hashlib.sha256(source.read_bytes()).hexdigest()==tables['sha256']
binding=json.loads((HERE/'vendor_output_binding.json').read_text())
def validate_binding(cubin_hash,cfg_hash,sass_hash,binding):
    assert cubin_hash==binding['cubin_sha256'], 'Stale cubin'
    assert cfg_hash==binding['cfg_sha256'], 'Stale CFG'
    assert sass_hash==binding['sass_normalized_text_sha256'], 'Stale SASS'
def validate_inventory(functions,expected):
    observed={name:sum(bool(x.gmma and x.gmma['rs']) for x in ins) for name,ins in functions.items() if any(x.gmma for x in ins)}
    assert expected and observed==expected and sum(observed.values())>0, ('Missing or unexpected RS inventory',observed,expected)
    return observed
validate_binding(hashlib.sha256(source.read_bytes()).hexdigest(),hashlib.sha256(cfg.encode()).hexdigest(),hashlib.sha256(sass.encode()).hexdigest(),binding)
validate_inventory(mod.parse_sass(sass),binding['expected_rs_inventory'])
nodes={};lines=cfg.splitlines()
for i,line in enumerate(lines):
 if line.startswith('[label=') and i and re.fullmatch(r'"[^"]+"',lines[i-1]):
  name=lines[i-1][1:-1];offsets=[int(x,16) for x in re.findall(r'([0-9a-f]{4,}):\\',line)]
  if offsets:nodes[name]={'first':offsets[0],'last':offsets[-1],'body':line}
branches={};proof=[]
for name,node in nodes.items():
 if 'BRX\\ ' not in node['body']:continue
 m=re.search(r'BRANCH_TARGETS\\ ([.A-Za-z0-9_,]+)',node['body']);assert m,('Missingvendortargets',name)
 targets=m.group(1).split(',');addresses=sorted(set(nodes[x]['first'] for x in targets))
 edges=set(re.findall(re.escape('"'+name+'"')+r':[^\n]*? -> "([^"]+)"',cfg))
 assert set(targets)==edges,('Annotation/CFGedge mismatch',name,targets,edges)
 branches[node['last']]=addresses;proof.append({'source_node':name,'address':hex(node['last']),'target_labels':targets,'target_addresses':[hex(a) for a in addresses]})
assert len(branches)==4
words=set(int(x,16) for s in tables['sections'] if s['name'].startswith('.nv.constant2.') for x in s['words'])
assert set(a for targets in branches.values() for a in targets)==words
native_link=mod._link
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
raw_edge_count=cfg.count(' -> ')
assert raw_edge_count==sum(map(len,vendor_edges.values())),('UnparsedorduplicateDOTedge',raw_edge_count,sum(map(len,vendor_edges.values())))
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
 # nvdisasm omits a disconnected terminal self-BRA trap and alignment NOPs.
 # Accept only this exact instruction class, with no incoming covered edge;
 # retain native links and prove entry cannot reach any omitted instruction.
 assert all(a>max(cfg_offsets) for a in omitted),('UncoveredinteriorSASS',omitted)
 for address in omitted:
  pad=ins[idx[address]]
  assert pad.op=='NOP' or (pad.op=='BRA' and not pad.pred and pad.text==f'BRA {hex(address)}'),('UncoverednonpaddingSASS',pad.text)
 assert not any(ins[j].addr in omitted for x in ins if x.addr not in omitted for j in x.succ),'Coverededgeintoomittedtail'
 observed=set();changes=[]
 for i,x in enumerate(ins):
  base=x.op.split('.')[0]
  if base in ('BRX','JMX'):
   assert base=='BRX' and x.addr in branches and not x.pred,('Unknownindirectbranch',x.text)
   m=re.fullmatch(r'BRX R[0-9]+ -0x([0-9a-f]+)',x.text);assert m and int(m.group(1),16)==x.addr+16
   assert set(branches[x.addr])==vendor_edges[x.addr]
   observed.add(x.addr)
  elif base in ('BRA','JMP','CALL'):
   targets=[int(m.group(1),16) for o in x.ops if (m:=re.match(r'^`?\(?(0x[0-9a-f]+)',o))]
   assert len(targets)==1 and targets[0] in idx,('Unresolveddirectcontrol',x.text)
  if base in ('BRX','JMX','BRA','JMP','CALL','RET','EXIT','BPT','KILL'):
   assert x.addr in vendor_edges or x.addr in omitted,('Controlnotavendorblockterminal',x.text)
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
 assert not any(ins[i].addr in omitted for i in seen),'Reachableomittedtail'
 assert set(cfg_offsets)<={ins[i].addr for i in seen},'Unreachablevendorinstruction'
 link_proof.update(vendor_blocks=len(vendor_edges),vendor_edges=sum(map(len,vendor_edges.values())),raw_dot_edges=raw_edge_count,all_vendor_instructions_reachable=True,all_vendor_edges_covered=True,
  omitted_unreachable_tail_instructions=len(omitted),reachable_rs=len(all_rs),repaired_successors=changes,call_helpers=helpers)
 return True

report={'scope':'Upstream RS/accumulator state analysis unchanged; all vendor block-terminal edges covered, four BRX targets matched to exact cubin constant2 table, and call helpers verified GMMA-state-neutral. Static screen, not hardware correctness.','checker_sha256':hashlib.sha256(pathlib.Path(spec.origin).read_bytes()).hexdigest(),'sass_sha256':hashlib.sha256(sass.encode()).hexdigest(),'cfg_sha256':hashlib.sha256(cfg.encode()).hexdigest(),'cubin_sha256':tables['sha256'],'branch_evidence':proof,'complete':False,'functions':[]}
def save():(HERE/'finite_vendor_cfg_audit.json').write_text(json.dumps(report,indent=2)+'\n')
save();mod._link=resolved_link;start=time.monotonic()
for name,ins in mod.parse_sass(sass).items():
 if not any(x.gmma for x in ins):continue
 rs,problems,complete=mod.check_function(ins);row={'name':name,'instructions':len(ins),'rs':rs,'analysis_complete':complete,'problem_count':len(problems),'first_problems':problems[:20]};report['functions'].append(row);save();print(json.dumps(row),flush=True)
 # Negative control: inject an ordinary write to the first pending A register,
 # without changing instruction addresses or branch successors.
 neg=mod.parse_sass(sass)[name]
 first=next(i for i,x in enumerate(neg) if x.gmma and x.gmma['rs']);a=min(neg[first].gmma['a']);at=first+1
 assert neg[at].op.split('.')[0] not in ('BRA','BRX','JMX','CALL','JMP','RET','EXIT')
 neg[at]=mod._Ins(neg[at].addr,f'MOV R{a}, RZ')
 nrs,nproblems,ncomplete=mod.check_function(neg)
 row['negative_injected_write']={'address':hex(neg[at].addr),'register':a,'rs':nrs,'analysis_complete':ncomplete,'problem_count':len(nproblems),'detected':any('A-write' in x for x in nproblems)};save()
assert len(report['functions'])==len(binding['expected_rs_inventory']) and sum(x['rs'] for x in report['functions'])==sum(binding['expected_rs_inventory'].values())
report['cfg_coverage']=dict(link_proof)
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

report['artifact_binding_sha256']=hashlib.sha256((HERE/'vendor_output_binding.json').read_bytes()).hexdigest()
report.update(complete=True,seconds=time.monotonic()-start,all_analyses_complete=all(x['analysis_complete'] for x in report['functions']),problem_count=sum(x['problem_count'] for x in report['functions']),negative_controls_detected=all(x['negative_injected_write']['detected'] for x in report['functions']));save();print(json.dumps({k:v for k,v in report.items() if k not in('functions','branch_evidence')}),flush=True)

report['passed']=bool(report['complete'] and report['all_analyses_complete'] and report['problem_count']==0 and report['negative_controls_detected'] and report['negative_omitted_bra_div_edge']['detected'] and report['cfg_coverage']['all_vendor_edges_covered']);save()
assert report['passed'],'StaticvendorCFGhazardgatefailed'
