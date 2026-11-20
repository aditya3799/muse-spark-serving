import pathlib,re,json,hashlib,resource,importlib.util,time,sys,collections
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('hazard',HERE/'python__sglang__kernels__jit__utils__wgmma_rs_hazard.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
name=sys.argv[1];assert name in ('pr50800_offline','pr50800_boundaries')
folder=HERE/name; source=HERE.parents[1]/'moe'/name; binding=json.loads((folder/'vendor_binding.json').read_text())
cubin=source/('pr50800_sm90a.cubin' if name=='pr50800_offline' else 'boundary_sm90a.cubin')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(cubin)==binding['cubin_sha256']
for item in binding['tools']:
 assert sha(folder/item['output'])==item['output_sha256']
 assert sha(pathlib.Path(item['command'][0]))==item['tool_sha256']
sass=(folder/'sass_recheck.txt').read_text();assert sass==(source/'sass.txt').read_text()
cfg=(folder/'vendor_bbcfg.dot').read_text();lines=cfg.splitlines();nodes={};fn=None
for i,line in enumerate(lines):
 m=re.fullmatch(r'subgraph "cluster_(.+)" \{',line)
 if m:fn=m.group(1)
 if line.startswith('[label=') and i and re.fullmatch(r'"[^"]+"',lines[i-1]):
  label=lines[i-1][1:-1];offsets=[int(x,16) for x in re.findall(r'([0-9a-f]{4,}):\\',line)]
  if offsets:
   assert label not in nodes
   nodes[label]={'fn':fn,'first':offsets[0],'last':offsets[-1],'offsets':offsets,'body':line}
edges={k:set() for k in nodes}
parsed_edges=re.findall(r'^"([^"]+)":[^\n]*? -> "([^"]+)"',cfg,re.M)
raw_edge_count=sum(' -> ' in line for line in lines)
assert len(parsed_edges)==raw_edge_count,'Unparsed vendor edge'
assert sum(line.startswith('[label=') for line in lines)==len(nodes),'Unparsed vendor node'
for a,b in parsed_edges:
 assert a in nodes and b in nodes,(a,b)
 assert nodes[a]['fn']==nodes[b]['fn'],('Crossfunctionedge',a,b)
 edges[a].add(nodes[b]['first'])
functions=mod.parse_sass(sass)
expected={n:sum(bool(x.gmma and x.gmma['rs']) for x in ins) for n,ins in functions.items() if any(x.gmma for x in ins)}
assert len(expected)==2 and sorted(expected.values())==[160,224],expected
native_link=mod._link; current_name=None; last_coverage={}

def validate_inventory(fn,ins,checked_nodes,checked_edges):
 selected={k:v for k,v in checked_nodes.items() if v['fn']==fn}
 assert selected,'Missing function nodes'
 occurrences=collections.Counter(a for v in selected.values() for a in v['offsets'])
 by_address={x.addr:x for x in ins};addresses=set(by_address);covered=set(occurrences)
 assert not covered-addresses and all(n==1 for n in occurrences.values()),'Invalid/duplicate instruction coverage'
 assert all(v['offsets']==list(range(v['first'],v['last']+16,16)) for v in selected.values()),'Noncontiguous block'
 # These exact two artifacts have a disconnected terminal self-branch plus NOP
 # alignment island. No other absent instruction or unreachable block is allowed.
 trailing=[x for x in ins if x.addr not in covered]
 expected_tail={('pr50800_offline',160):9,('pr50800_offline',224):11,
                ('pr50800_boundaries',160):14,('pr50800_boundaries',224):15}[(name,expected[fn])]
 assert len(trailing)==expected_tail,'Unexpected omitted instruction count'
 assert [x.addr for x in trailing]==list(range(max(covered)+16,ins[-1].addr+16,16)),'Nonterminal omission'
 first=trailing[0]
 assert first.op=='BRA' and not first.pred and first.text==f'BRA {first.addr:#x}','Invalid terminal island head'
 assert all(x.op=='NOP' and not x.pred for x in trailing[1:]),'Invalid terminal padding'
 assert native_link(ins),'Unresolved original control'
 assert all(ins[j].addr in covered for x in ins if x.addr in covered for j in x.succ),'Reachable omitted instruction'
 for v in selected.values():
  for a in v['offsets'][:-1]:
   x=by_address[a]
   assert {ins[j].addr for j in x.succ}=={a+16},'Unmodeled intrablock control'
 starts={v['first']:k for k,v in selected.items()};stack=[starts[ins[0].addr]];seen=set()
 while stack:
  key=stack.pop()
  if key in seen:continue
  seen.add(key)
  for target in checked_edges[key]:
   assert target in starts,'Cross-function or interior target'
   stack.append(starts[target])
 assert seen==set(selected),'Unreachable vendor block'
 reachable={a for k in seen for a in selected[k]['offsets']}
 assert all(x.addr in reachable for x in ins if x.gmma),'Unreachable GMMA'
 return {'vendor_instructions':len(covered),'vendor_blocks':len(selected),'reachable_blocks':len(seen),
         'all_gmma_reachable':True,'terminal_island_instructions':len(trailing),'terminal_island_head':hex(first.addr)}

inventory={fn:validate_inventory(fn,ins,nodes,edges) for fn,ins in functions.items() if fn in expected}
inventory_negatives={}
for fn in expected:
 selected=[k for k,v in nodes.items() if v['fn']==fn]
 checks={}
 for case in ('missing_node','missing_all_entry_edges','corrupt_trailing_island'):
  ni=mod.parse_sass(sass)[fn];nn=dict(nodes);ee={k:set(v) for k,v in edges.items()}
  if case=='missing_node':del nn[selected[len(selected)//2]]
  elif case=='missing_all_entry_edges':ee[next(k for k in selected if nodes[k]['first']==ni[0].addr)]=set()
  else:ni[-1]=mod._Ins(ni[-1].addr,'MOV R0, RZ')
  try:validate_inventory(fn,ni,nn,ee)
  except (AssertionError,KeyError):checks[case]=True
  else:checks[case]=False
 assert all(checks.values()),checks
 inventory_negatives[fn]=checks
def resolve(ins):
 native_complete=native_link(ins);assert native_complete,'Unresolvedindirectcontrol'
 idx={x.addr:i for i,x in enumerate(ins)};coverage=[]
 # This donor's two GMMA functions must be leaf routines. The one unrelated
 # device helper CALL is outside them; reject instead of assuming call effects.
 assert not any(x.op.split('.')[0] in ('CALL','RET','BRX','JMX') for x in ins)
 for label,node in nodes.items():
  if node['fn']!=current_name:continue
  assert all(a in idx for a in node['offsets'])
  at=idx[node['last']];x=ins[at];wanted=edges[label]
  assert all(a in idx for a in wanted)
  before={ins[j].addr for j in x.succ};missing=wanted-before
  if missing:
   assert x.op=='BRA.DIV' and not x.pred,('Unexpectedmissingvendorpath',x.text,missing)
   assert missing=={x.addr+16} and len(wanted)==2,(x.text,missing,wanted)
   x.succ=sorted(set(x.succ)|{idx[a] for a in wanted})
   x.back=any(ins[j].addr<=x.addr for j in x.succ)
  assert wanted.issubset({ins[j].addr for j in x.succ})
  coverage.append({'last':hex(x.addr),'op':x.op,'vendor_edges':[hex(a) for a in sorted(wanted)],'added_edges':[hex(a) for a in sorted(missing)]})
 assert coverage
 last_coverage[current_name]=coverage
 return True
mod._link=resolve
out={'complete':False,'scope':'Static RS/accumulator screen with all exact vendor block-terminal edges covered; no hardware correctness or timing. Earlier direct-only reports superseded.','cubin_sha256':sha(cubin),'vendor_binding_sha256':sha(folder/'vendor_binding.json'),'checker_sha256':sha(pathlib.Path(spec.origin)),'adapter_sha256':sha(pathlib.Path(__file__)),'expected_rs_inventory':expected,'raw_vendor_edge_count':raw_edge_count,'parsed_vendor_edge_count':len(parsed_edges),'coverage_inventory':inventory,'coverage_negative_controls':inventory_negatives,'functions':[]}
def save():(folder/'complete_cfg_audit.json').write_text(json.dumps(out,indent=2)+'\n')
save();start=time.monotonic()
try:
 for current_name,count in expected.items():
  ins=functions[current_name];rs,problems,complete=mod.check_function(ins)
  row={'name':current_name,'rs':rs,'analysis_complete':complete,'problem_count':len(problems),'first_problems':problems[:20],'vendor_blocks':len(last_coverage[current_name]),'repaired_branches':[x for x in last_coverage[current_name] if x['added_edges']]}
  assert rs==count
  neg=mod.parse_sass(sass)[current_name];first=next(i for i,x in enumerate(neg) if x.gmma and x.gmma['rs']);a=min(neg[first].gmma['a']);at=first+1
  assert neg[at].op.split('.')[0] not in ('BRA','BRX','JMX','CALL','JMP','RET','EXIT')
  neg[at]=mod._Ins(neg[at].addr,f'MOV R{a}, RZ');nrs,issues,ncomplete=mod.check_function(neg)
  row['negative_pending_a_write_detected']=any('A-write' in x for x in issues)
  row['negative_analysis_complete']=ncomplete
  # Original adapter loses at least one of the independently established
  # fallthrough edges. The new coverage requirement must reject it.
  raw=mod.parse_sass(sass)[current_name];native_link(raw);idx={x.addr:i for i,x in enumerate(raw)}
  row['negative_old_cfg_rejected']=any(not edges[label].issubset({raw[j].addr for j in raw[idx[node['last']]].succ}) for label,node in nodes.items() if node['fn']==current_name)
  out['functions'].append(row);save();print(json.dumps(row),flush=True)
 assert len(out['functions'])==2 and sum(x['rs'] for x in out['functions'])==384
 out.update(complete=True,seconds=time.monotonic()-start,all_analyses_complete=all(x['analysis_complete'] for x in out['functions']),problem_count=sum(x['problem_count'] for x in out['functions']),negative_controls_detected=all(x['negative_pending_a_write_detected'] and x['negative_old_cfg_rejected'] for x in out['functions']))
 out['static_screen_pass']=out['complete'] and out['all_analyses_complete'] and out['problem_count']==0 and out['negative_controls_detected']
except Exception as e:out['error']=type(e).__name__+': '+str(e)
save();print(json.dumps({k:v for k,v in out.items() if k!='functions'}),flush=True)
