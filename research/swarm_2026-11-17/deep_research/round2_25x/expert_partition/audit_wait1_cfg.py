from pathlib import Path
import re,json,hashlib,collections
P=Path(__file__).resolve().parent;H=P.parent/'hopper_moe';base=H/'offline_wait1_01/wait1'
def parse(path):
 instructions=[];labels={};pending=[]
 for line in path.read_text().splitlines():
  lab=re.match(r'\s*(\.L_x_\d+):',line)
  if lab:pending.append(lab.group(1))
  m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:(@!?P\d+)\s+)?([A-Z][A-Za-z0-9_.]*)\s*(.*?);',line)
  if not m:continue
  for name in pending:labels[name]=len(instructions)
  pending=[];addr,pred,op,args=m.groups();instructions.append((addr,pred,op,args))
 return instructions,labels

def audit(path,remove_final_wait=False):
 ins,labels=parse(path);start=next(i for i,x in enumerate(ins) if x[2].startswith('HGMMA.'))
 queue=collections.deque([(start,(),())]);seen=set();hazards=[];ends=0;edges=0
 while queue:
  pc,pending,uncommitted=queue.popleft();state=(pc,pending,uncommitted)
  if state in seen:continue
  seen.add(state)
  if len(seen)>200000:raise RuntimeError('state bound exceeded')
  if pc<0 or pc>=len(ins):raise RuntimeError('bad PC')
  addr,pred,op,args=ins[pc];nxt=pc+1
  if op.startswith('HGMMA.'):
   n=int(re.search(r'64x(\d+)x',op).group(1));regs=list(map(int,re.findall(r'\bR(\d+)\b',args)));dst,src=regs[:2]
   uncommitted+=((tuple(range(src,src+4)),tuple(range(dst,dst+n//2))),)
   if 'gsb0' in args:
    group=(tuple(sorted(set(r for g in uncommitted for r in g[0]))),tuple(sorted(set(r for g in uncommitted for r in g[1]))));pending+=(group,);uncommitted=()
  elif op=='WARPGROUP.DEPBAR.LE':
   count=int(re.search(r', 0x([0-9a-f]+)',args).group(1),16)
   if count==0 and not remove_final_wait:
    ends+=1;continue
   if count!=0:pending=pending[-count:]
  elif op.startswith('BRA'):
   target=re.search(r'(\.L_x_\d+)',args)
   if not target:raise RuntimeError('unknown branch '+str(ins[pc]))
   dest=labels[target.group(1)];queue.append((dest,pending,uncommitted));edges+=1
   if not pred:continue
  elif op=='EXIT':
   if pending or uncommitted:hazards.append({'address':addr,'kind':'exit_pending'})
   continue
  elif pending or uncommitted:
   alive=pending+uncommitted;inputs=set(r for g in alive for r in g[0]);accum=set(r for g in alive for r in g[1]);used=set(map(int,re.findall(r'\bR(\d+)\b',args)));dest=re.match(r'R(\d+)\b',args)
   if dest:
    width=4 if '.128' in op else 2 if '.64' in op or '.WIDE' in op or op=='CS2R' else 1
    written=set(range(int(dest.group(1)),int(dest.group(1))+width))
    if written&inputs:hazards.append({'address':addr,'kind':'pending_input_overwrite','regs':sorted(written&inputs),'instruction':op+' '+args})
   if used&accum:hazards.append({'address':addr,'kind':'pending_C_access','regs':sorted(used&accum),'instruction':op+' '+args})
  if remove_final_wait and hazards:return {'states':len(seen),'hazards':hazards[:5],'negative_detected':True}
  if len(pending)>3:raise RuntimeError('unexpected pending groups')
  queue.append((nxt,pending,uncommitted))
 return {'states':len(seen),'branch_transitions':edges,'drain_end_states':ends,'hazards':hazards,'negative_detected':False}
rows=[]
for path in sorted(base.glob('*wait1/kernel.sass')):
 good=audit(path);bad=audit(path,True);assert not good['hazards'] and good['drain_end_states'] and bad['negative_detected']
 rows.append({'job':path.parent.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'positive':good,'removed_final_drain_negative':bad})
r={'status':'BOUNDED_CONTROL_FLOW_REGISTER_CHECK_PASS','scope':'Independent CPU SASS branch-edge state propagation from first HGMMA through final drain. Conditional branch paths both explored; ordinary predicated writes conservatively treated as executed. Not hardware/complete ISA proof.','rows':rows,'limits':['Decoder only handles observed instruction syntax and known destination widths; implicit operands are not modeled.','WGMMA input-register read permission and repeated accumulator update are assumed per official PTX contract.','Shared descriptor validity and cross-warpgroup shared-A lifetime rely on separate source/event model.','Stops at final wait0; outer task recurrence/epilogue validated by source review, not this register model.','No H200 arithmetic, runtime, racecheck or throughput claim.']}
(P/'WAIT1_INDEPENDENT_CFG_SASS_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps([{'job':x['job'],'positive':x['positive'],'negative':x['removed_final_drain_negative']} for x in rows]))
