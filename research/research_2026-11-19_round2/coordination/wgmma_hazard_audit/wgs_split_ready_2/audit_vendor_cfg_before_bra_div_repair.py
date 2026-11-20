import pathlib,re,json,hashlib,resource,importlib.util,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('hazard',HERE/'python__sglang__kernels__jit__utils__wgmma_rs_hazard.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
cfg=(HERE/'finite_vendor_bbcfg.dot').read_text();sasspath=pathlib.Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/moe/wgs_split_ready_2_sass.txt');sass=sasspath.read_text()
receipt=json.loads((HERE/'vendor_bbcfg_receipt.json').read_text());tables=json.loads((HERE/'finite_constant_tables.json').read_text())
assert receipt['exit_code']==0 and receipt['cubin_sha256']==tables['sha256']
source=pathlib.Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/moe/wgs_split_ready_2_sm90a.cubin')
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

def resolved_link(ins):
 native_link(ins);idx={x.addr:i for i,x in enumerate(ins)}
 observed=set()
 for i,x in enumerate(ins):
  base=x.op.split('.')[0]
  if base in ('BRX','JMX'):
   assert base=='BRX' and x.addr in branches and not x.pred,('Unknownindirectbranch',x.text)
   m=re.fullmatch(r'BRX R[0-9]+ -0x([0-9a-f]+)',x.text);assert m and int(m.group(1),16)==x.addr+16
   assert all(a in idx for a in branches[x.addr])
   x.succ=[idx[a] for a in branches[x.addr]];x.back=any(a<=x.addr for a in branches[x.addr]);observed.add(x.addr)
  elif base in ('BRA','JMP','CALL'):
   targets=[int(m.group(1),16) for o in x.ops if (m:=re.match(r'^`?\(?(0x[0-9a-f]+)',o))]
   assert len(targets)==1 and targets[0] in idx,('Unresolveddirectcontrol',x.text)
 assert observed==set(branches)
 return True

report={'scope':'Upstream RS/accumulator state analysis unchanged; only four BRX successors resolved from NVIDIA vendor CFG and exact cubin constant2 table. Static screen, not hardware correctness.','checker_sha256':hashlib.sha256(pathlib.Path(spec.origin).read_bytes()).hexdigest(),'sass_sha256':hashlib.sha256(sass.encode()).hexdigest(),'cfg_sha256':hashlib.sha256(cfg.encode()).hexdigest(),'cubin_sha256':tables['sha256'],'branch_evidence':proof,'complete':False,'functions':[]}
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
report['artifact_binding_sha256']=hashlib.sha256((HERE/'vendor_output_binding.json').read_bytes()).hexdigest()
report.update(complete=True,seconds=time.monotonic()-start,all_analyses_complete=all(x['analysis_complete'] for x in report['functions']),problem_count=sum(x['problem_count'] for x in report['functions']),negative_controls_detected=all(x['negative_injected_write']['detected'] for x in report['functions']));save();print(json.dumps({k:v for k,v in report.items() if k not in('functions','branch_evidence')}),flush=True)
