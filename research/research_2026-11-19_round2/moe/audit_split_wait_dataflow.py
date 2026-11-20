"""Classify first waits vs retry loops and trace decoded-value sleep dependency."""
import re,json,hashlib
from pathlib import Path
here=Path(__file__).resolve().parent
reports=[]
for name in ('split_ready','split_dependency'):
 p=here/('wgs_'+name+'_2_sass.txt');text=p.read_text();ins=[]
 for line in text.splitlines():
  m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?);',line)
  if m:ins.append((int(m.group(1),16),m.group(2).strip()))
 address={a:i for i,(a,_) in enumerate(ins)};rows=[]
 for i,(a,line) in enumerate(ins):
  if 'SYNCS.PHASECHK.TRANS64.TRYWAIT' not in line or '+0x30]' not in line:continue
  pred=re.search(r'TRYWAIT (P\d+)',line).group(1)
  end=next(j for j in range(i+1,min(i+200,len(ins))) if '@!'+pred+' BRA ' in ins[j][1])
  target=int(re.search(r'BRA 0x([0-9a-f]+)',ins[end][1]).group(1),16)
  if target<=ins[end][0]:continue # Retry loop, not a separate consumer stage site.
  prev=max(j for j in range(i) if 'SYNCS.PHASECHK.TRANS64.TRYWAIT' in ins[j][1])
  mul=[x for _,x in ins[prev+1:end] if x.startswith('HMUL2.BF16')]
  final=[x for x in mul if '8.50705917302346158658e+37' not in x]
  regs=[re.match(r'HMUL2.BF16\S* (R\d+)',x).group(1) for x in final]
  traced=set();sleep=None
  if name=='split_dependency':
   taint={r:{r} for r in regs}
   for _,x in ins[address[target]:address[target]+40]:
    m=re.match(r'LOP3.LUT (R\d+), (.*)',x)
    if m:taint[m.group(1)]=set().union(*(taint.get(r,set()) for r in re.findall(r'\bR\d+\b',m.group(2))))
    m=re.search(r'NANOSLEEP.SYNCS (R\d+)',x)
    if m:traced=taint.get(m.group(1),set());sleep=x;break
   assert len(regs)==8 and traced==set(regs),(hex(a),regs,traced)
  rows.append(dict(first_check=hex(a),retry_target=hex(target),bf16_multiplications_before_retry_branch=len(mul),decoded_pair_outputs=len(regs),sleep_instruction=sleep,decoded_pairs_consumed_by_sleep_hint=len(traced)))
 assert len(rows)==32
 reports.append(dict(variant=name,sass_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),first_wait_sites=len(rows),retry_wait_sites=32,first_sites_without_bf16_multiplies=sum(x['bf16_multiplications_before_retry_branch']==0 for x in rows),all_first_sites_full_group_before_retry=all(x['bf16_multiplications_before_retry_branch']==16 for x in rows),all_dependency_hints_consume8pairs=all(x['decoded_pairs_consumed_by_sleep_hint']==8 for x in rows) if name=='split_dependency' else None,rows=rows))
r=dict(scope='Static SASS dataflow. First TRYWAIT can be scheduled early; all decoded pair values precede the branch into the dependency variant retry loop and feed NANOSLEEP.SYNCS. This is not proof that first hardware TRYWAIT never suspends, nor any latency/progress evidence.',reports=reports)
(here/'split_wait_dataflow.json').write_text(json.dumps(r,indent=2))
print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in reports],indent=2))
