"""Independent parser coverage check; no GPU, not an arithmetic proof."""
import collections, hashlib, importlib.util, json, pathlib, re
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parent/'coordination'/'wgmma_hazard_audit'
spec=importlib.util.spec_from_file_location('checker',ROOT/'python__sglang__kernels__jit__utils__wgmma_rs_hazard.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
out={}
for name in ['pr50800_offline','pr50800_boundaries']:
 folder=ROOT/name; text=(folder/'vendor_bbcfg.dot').read_text(); lines=text.splitlines(); nodes={}; fn=None
 for i,line in enumerate(lines):
  m=re.fullmatch(r'subgraph "cluster_(.+)" \{',line)
  if m:fn=m.group(1)
  if line.startswith('[label=') and i and re.fullmatch(r'"[^"]+"',lines[i-1]):
   offsets=[int(x,16) for x in re.findall(r'([0-9a-f]{4,}):\\',line)]
   if offsets:nodes[lines[i-1][1:-1]]=(fn,offsets)
 parsed_edges=re.findall(r'^"([^"]+)":[^\n]*? -> "([^"]+)"',text,re.M)
 raw_edges=[x for x in lines if ' -> ' in x]
 assert len(parsed_edges)==len(raw_edges)
 funcs=mod.parse_sass((folder/'sass_recheck.txt').read_text()); result=[]
 for fn,ins in funcs.items():
  if not any(x.gmma for x in ins):continue
  selected={k:v[1] for k,v in nodes.items() if v[0]==fn}
  occurrences=collections.Counter(a for offsets in selected.values() for a in offsets)
  addresses={x.addr for x in ins}; covered=set(occurrences)
  # cuobjdump may include post-EXIT alignment NOP padding; record rather than hide it.
  missing=[{'addr':hex(x.addr),'instruction':x.text} for x in ins if x.addr not in covered]
  assert not covered-addresses
  assert all(n==1 for n in occurrences.values())
  # Independently traverse vendor block successors from the true first instruction.
  start={v[0]:k for k,v in selected.items()}; edges={k:[] for k in selected}
  for a,b in parsed_edges:
   if a in selected:
    assert b in selected;edges[a].append(b)
  stack=[start[ins[0].addr]];seen=set()
  while stack:
   node=stack.pop()
   if node in seen:continue
   seen.add(node);stack.extend(edges[node])
  reachable={a for k in seen for a in selected[k]}
  unreachable_rs=[hex(x.addr) for x in ins if x.gmma and x.gmma['rs'] and x.addr not in reachable]
  result.append(dict(function=fn,instructions=len(ins),vendor_instructions=len(covered),blocks=len(selected),reachable_blocks=len(seen),uncovered=missing,unreachable_rs=unreachable_rs))
 out[name]=dict(raw_edges=len(raw_edges),parsed_edges=len(parsed_edges),functions=result,receipt_sha256=hashlib.sha256((folder/'complete_cfg_audit.json').read_bytes()).hexdigest())
(HERE/'donor_cfg_independent_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
