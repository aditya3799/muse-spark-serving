from pathlib import Path
import re,json
p=Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/coordination/wgmma_hazard_audit/finite_vendor_bbcfg.dot');s=p.read_text();lines=s.splitlines();nodes={}
for i,line in enumerate(lines):
 if line.startswith('[label=') and i and re.fullmatch(r'"[^"]+"',lines[i-1]):
  offsets=[int(x,16) for x in re.findall(r'([0-9a-f]{4,}):\\',line)]
  if offsets:nodes[lines[i-1][1:-1]]=(offsets[0],offsets[-1],line)
for name,(first,last,line) in nodes.items():
 if 'BRA.DIV' not in line:continue
 targets=re.findall(re.escape('"'+name+'"')+r':[^\n]*? -> "([^"]+)"',s)
 print(json.dumps(dict(node=name,first=hex(first),last=hex(last),targets=[dict(name=t,first=hex(nodes[t][0])) for t in targets])))
