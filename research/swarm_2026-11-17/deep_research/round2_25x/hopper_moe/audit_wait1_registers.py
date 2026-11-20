from pathlib import Path
import re,json
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');R=P/'offline_wait1_01/wait1';rows=[]
for d in sorted(R.iterdir()):
 if not d.is_dir() or not d.name.endswith('wait1'):continue
 lines=(d/'kernel.sass').read_text().splitlines();pending=[];uncommitted=[];hazards=[];waits=[];groups=[];active=False
 for line in lines:
  m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Za-z0-9_.]*)\s*(.*?);',line)
  if not m:continue
  addr,op,args=m.groups()
  if op.startswith('HGMMA.'):
   active=True;n=int(re.search(r'64x(\d+)x',op).group(1));regs=[int(x) for x in re.findall(r'\bR(\d+)\b',args)];dst,operand=regs[:2];g={'inputs':set(range(operand,operand+4)),'accum':set(range(dst,dst+n//2))};uncommitted.append(g)
   if 'gsb0' in args:
    item={'inputs':set().union(*(x['inputs'] for x in uncommitted)),'accum':set().union(*(x['accum'] for x in uncommitted))};pending.append(item);groups.append({'commit_address':addr,'input_registers':sorted(item['inputs']),'accumulator_registers':sorted(item['accum'])});uncommitted=[]
   continue
  if op=='WARPGROUP.DEPBAR.LE':
   n=int(re.search(r', 0x([0-9a-f]+)',args).group(1),16);waits.append({'address':addr,'remaining':n})
   while len(pending)>n:pending.pop(0)
   continue
  if not active or not pending and not uncommitted:continue
  alive=pending+uncommitted;inputs=set().union(*(x['inputs'] for x in alive));accum=set().union(*(x['accum'] for x in alive));used=set(int(x) for x in re.findall(r'\bR(\d+)\b',args));dest=re.match(r'R(\d+)\b',args)
  if dest:
   width=4 if '.128' in op else 2 if '.64' in op or '.WIDE' in op or op=='CS2R' else 1
   written=set(range(int(dest.group(1)),int(dest.group(1))+width))
   if written&inputs:hazards.append({'address':addr,'kind':'pending_input_overwrite','registers':sorted(written&inputs),'instruction':line.strip()})
  if used&accum:hazards.append({'address':addr,'kind':'pending_accumulator_ordinary_access','registers':sorted(used&accum),'instruction':line.strip()})
 rows.append({'job':d.name,'committed_groups':len(groups),'wait1_count':sum(x['remaining']==1 for x in waits),'wait0_count':sum(x['remaining']==0 for x in waits),'final_pending_groups':len(pending),'hazards':hazards,'groups':groups,'waits':waits})
r={'status':'LINEAR_MAINLOOP_REGISTER_AUDIT','scope':'Static destination width/accumulator use check between WGMMA commit/waits; includes all unrolled stages. Not full CFG or hardware race proof. Shared A lifetime rests on separately reviewed barrier model.','rows':rows};(P/'WAIT1_REGISTER_LIFETIME_SASS.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps([{'job':x['job'],'groups':x['committed_groups'],'wait1':x['wait1_count'],'wait0':x['wait0_count'],'hazards':x['hazards']} for x in rows]))

