from pathlib import Path
import re,json,importlib.util
p=Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/coordination/wgmma_hazard_audit');s=(p/'finite_vendor_bbcfg.dot').read_text();lines=s.splitlines();nodes={}
sp=importlib.util.spec_from_file_location('hazard',p/'python__sglang__kernels__jit__utils__wgmma_rs_hazard.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ins=next(iter(m.parse_sass((p.parent.parent/'moe/wgs_finite_2_sass.txt').read_text()).values()));m._link(ins);idx={x.addr:i for i,x in enumerate(ins)}
for i,line in enumerate(lines):
 if line.startswith('[label=') and i and re.fullmatch(r'"[^"]+"',lines[i-1]):
  offsets=[int(x,16) for x in re.findall(r'([0-9a-f]{4,}):\\',line)]
  if offsets:nodes[lines[i-1][1:-1]]=(offsets,line)
alladdr=[a for offsets,_ in nodes.values() for a in offsets]
print(json.dumps(dict(instructions=len(ins),cfg_addresses=len(alladdr),unique=len(set(alladdr)),missing=[hex(x) for x in set(idx)-set(alladdr)][:15],extra=[hex(x) for x in set(alladdr)-set(idx)][:15])))
counts={};examples=[]
for name,(offsets,line) in nodes.items():
 targets=re.findall(re.escape('"'+name+'"')+r':[^\n]*? -> "([^"]+)"',s)
 vendor={nodes[t][0][0] for t in targets};x=ins[idx[offsets[-1]]];native={ins[i].addr for i in x.succ}
 if vendor!=native:
  counts[x.op]=counts.get(x.op,0)+1
  if len(examples)<30:examples.append(dict(addr=hex(x.addr),op=x.text,native=[hex(a) for a in sorted(native)],vendor=[hex(a) for a in sorted(vendor)]))
print(json.dumps(dict(counts=counts,examples=examples),indent=2))
