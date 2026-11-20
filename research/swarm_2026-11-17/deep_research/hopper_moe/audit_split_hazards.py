from pathlib import Path
import json,re,hashlib
P=Path(__file__).resolve().parent;data=json.loads((P/'SPLIT_ISSUE_SASS_AUDIT.json').read_text());total=0;negative=0;hazards=[]
def analyze(text,c_live,b_live):
 refs={int(x) for x in re.findall(r'\bR(\d+)\b',text)}
 c_access=sorted(refs&c_live)
 dest=re.match(r'\S+\s+R(\d+)\b',text);b_write=[]
 if dest:
  start=int(dest.group(1));op=text.split()[0];width=4 if '.128' in op else 2 if '.64' in op else 1
  b_write=sorted(set(range(start,start+width))&b_live)
 return c_access,b_write
for row in data['rows']:
 for w in row['windows']:
  cs=set();bs=set()
  for text in w['hgmmas']:
   regs=re.findall(r'\bR(\d+)\b',text);n=int(re.search(r'HGMMA\.64x(\d+)x16',text).group(1));a,b=int(regs[0]),int(regs[1]);cs.update(range(a,a+n//2));bs.update(range(b,b+4))
  for op in w['ops']:
   a,b=analyze(op['text'],cs,bs)
   if a or b:hazards.append({'job':row['job_id'],'instruction':op,'c_access':a,'b_write':b})
  assert analyze('MOV R255, R%d'%min(cs),cs,bs)[0]
  assert analyze('MOV R%d, RZ'%min(bs),cs,bs)[1]
  negative+=2;total+=1
assert not hazards,hazards[:2]
r={'status':'all_pending_windows_passed_with_hazard_negative_controls','pending_windows':total,'mutated_hazard_cases_caught':negative,'accumulator_access_or_operand_write_hazards':hazards,'scope':'Lexical explicit-register audit only; negative controls inject illegal scalar Creads and liveBoverwrites into actual decoded pending windows. Not fullscoreboard orhardware proof.','input_sha256':hashlib.sha256((P/'SPLIT_ISSUE_SASS_AUDIT.json').read_bytes()).hexdigest()}
(P/'SPLIT_ISSUE_HAZARD_NEGATIVE_CONTROLS.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
