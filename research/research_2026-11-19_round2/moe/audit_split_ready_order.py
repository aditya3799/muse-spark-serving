"""Inspect all activation-ready fast-path wait/branch windows; static evidence only."""
import re,json,hashlib
from pathlib import Path
here=Path(__file__).resolve().parent
p=here/'wgs_split_ready_2_sass.txt'
ins=[line.strip() for line in p.read_text().splitlines() if re.search(r'/\*[0-9a-f]+\*/',line)]
rows=[]
for i,line in enumerate(ins):
 if 'SYNCS.PHASECHK.TRANS64.TRYWAIT' not in line or '+0x30]' not in line:continue
 prev=max((j for j in range(i) if 'SYNCS.PHASECHK.TRANS64.TRYWAIT' in ins[j]),default=0)
 pred=re.search(r'TRYWAIT (P\d+)',line).group(1)
 end=next((j for j in range(i+1,min(i+200,len(ins))) if '@!'+pred+' BRA ' in ins[j]),None)
 assert end is not None,line
 window=ins[prev+1:end]
 rows.append(dict(activation_trywait=line,blocking_branch=ins[end],prior_wait=ins[prev],coefficient_bf16_multiplies=sum('HMUL2.BF16' in x for x in window),shared_loads=sum('LDS.' in x for x in window),integer_ops=sum(any(k in x for k in ('IMAD','LOP3','SHF.')) for x in window),window_instructions=end-prev-1))
r=dict(sass_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),sites=len(rows),sites_with_zero_bf16_multiplies=sum(x['coefficient_bf16_multiplies']==0 for x in rows),rows=rows,scope='Linear fast-path disassembly windows between previous readiness check and activation blocking branch. Outlined wait loops and runtime delays not modeled; no performance claim.')
(here/'split_ready_compiler_order.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
