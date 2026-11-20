from pathlib import Path
import json,hashlib,shutil
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');O=P.parents[1]/'hopper_moe';A=P/'h200_gate_assets';A.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(root,job):
 receipt=root/'results.json';r=json.loads(receipt.read_text());row=next(x for x in r['cases'] if x['job_id']==job);assert row['status']=='compiled';src=root/job
 assert sha(src/'kernel.sm90a.cubin')==row['cubin_sha256'];assert sha(src/'kernel.cu')==row['generated_sha256']
 return row,src,receipt
base=P/'offline_sm90_01/virtual';wait=P/'offline_wait1_01/wait1';pairs=[]
for j in json.loads((P/'sm90_corrected/virtual_k_jobs.json').read_text())['jobs']:
 if not j['policy'].startswith('virtual'):continue
 cid=j['job_id'];stage=j['stage'];m=j['tokens']
 if cid.startswith('w13_m8'):br,bid=wait,'w13_m8_s4_control'
 elif cid.startswith('w2_m8'):br,bid=O/'offline_split_issue','w2_m8_control'
 elif cid.startswith('w2_m128'):br,bid=O/'offline_split_issue','w2_m128_control'
 elif cid.startswith('w2_m32'):br,bid=base,'w2_m32_s3_control__writer_fixed'
 else:br,bid=base,'w13_m32_actual_control'
 pairs.append((cid,'virtual',stage,m,br,bid,base,cid))
for m in (8,32):
 for st in (3,4):
  stem='w13_m%d_s%d'%(m,st);pairs.append((stem+'_wait1','wait1','w13',m,wait,stem+'_control',wait,stem+'_wait1'))
# WM48 changes native opcode width; this is an experimental numerical falsification pair, never presumed exact.
pairs.append(('w13_m32_bm48_wm48_virtual','wm48','w13',32,base,'w13_m32_actual_control',P/'offline_wait1_01/wm48','w13_m32_bm48_wm48_virtual'))
rows=[]
for name,family,stage,m,br,bid,cr,cid in pairs:
 row={'job_id':name,'family':family,'stage':stage,'tokens':m,'qualification':'UNEXECUTED_H200_NUMERICAL_GATE','native_opcode_width_changed':family=='wm48'}
 for side,root,jid in [('baseline',br,bid),('candidate',cr,cid)]:
  v,src,rec=load(root,jid);dest=A/(name+'__'+side);dest.mkdir(exist_ok=True)
  for f in ('kernel.sm90a.cubin','kernel.cu'):shutil.copyfile(src/f,dest/f)
  row[side]={'configuration':v['configuration'],'cubin_path':str((dest/'kernel.sm90a.cubin').relative_to(P)).replace('\\','/'),'cubin_sha256':sha(dest/'kernel.sm90a.cubin'),'source_path':str((dest/'kernel.cu').relative_to(P)).replace('\\','/'),'source_sha256':sha(dest/'kernel.cu'),'original_receipt':str(rec),'original_receipt_sha256':sha(rec),'resources':v['resource_lines']}
  if side=='baseline':bc=v['configuration']
  else:
   assert all(v['configuration'][k]==bc[k] for k in ('shape_n','shape_k','num_experts','a_dtype','b_dtype','bs_dtype','c_dtype','weight_scale_group_size','weight_scale_type','weight_scale_2_type'))
 rows.append(row)
sourcefiles=[*sorted((P/'sm90_corrected').glob('virtual_k_*.cuh')),P/'sm90_corrected/virtual_k_codegen.py',*sorted((P/'wait1').glob('wait1_*.cuh')),P/'wait1/wait1_codegen.py',P/'configuration_contract.py']
r={'status':'PREPARED_NOT_EXECUTED','scope':'Finite direct indexed GEMM bitwise gate only. New corrected ownership source+cubins. No timing, submission, serving hooks or old held gate.','pairs':rows,'research_source_hashes':{str(f.relative_to(P)).replace('\\','/'):sha(f) for f in sourcefiles}}
(P/'H200_CORRECTED_GATE_MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print('Pinned',len(rows),'pairs into owned gate_assets')
