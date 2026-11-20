from pathlib import Path
import json,sys,re,collections
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));from cuda_elf_reader import cuda_elf_constants
rows=[]
for folder in ('offline_virtual_k','offline_warp_n64_broad','offline_occupancy_m8','offline_occupancy_cpasync','offline_small_k_repair'):
 report=json.loads((P/folder/'results.json').read_text())
 for case in report['cases']:
  d=P/folder/case['job_id'];elf=cuda_elf_constants(d/'kernel.sm90a.cubin');values=elf['constants']
  assert values['NUM_THREADS']==case['resolved']['total_threads']
  assert values['NUM_MATH_THREADS']==case['resolved']['math_threads']
  assert values['NUM_LOAD_THREADS']==case['resolved']['load_threads']
  row={'folder':folder,'job':case['job_id'],'compiled_constants':values,'static_shared_bytes':elf['static_shared_bytes'],'estimated_dynamic_shared_bytes':case['shared_bytes_estimated'],'compiled_total_shared_bytes':values['SMEM_SIZE']+elf['static_shared_bytes'],'two_CTA_shared_capacity_pass':2*(values['SMEM_SIZE']+elf['static_shared_bytes'])<=228*1024}
  if (d/'sass.txt').exists():
   text=(d/'sass.txt').read_text();ops=[]
   for line in text.splitlines():
    match=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*',line)
    if match:ops.append(re.sub(r'^@!?\w+\s+','',match.group(2)).split()[0])
   row['opcode_counts']=dict(collections.Counter(op for op in ops if op.startswith(('BAR','FADD','HGMMA','WARPGROUP.DEPBAR'))))
  rows.append(row)
r={'status':'ELF_constants_and_resource_audit_passed','scope':'Actual cubin constants and static shared sections; no GPU context or occupancy call. Capacity is not achieved occupancy.','findings':['ELF reports1024 bytes static shared allocation in addition to exported dynamic SMEM_SIZE; resource models must include both.','Source-estimated shared bytes can differ from actual compiled SMEM_SIZE; compiled constants take precedence.','All tested exported launch-thread constants match intended physical threads.'],'rows':rows}
(P/'CUDA_ELF_RESOURCE_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n')
for row in rows:
 if row['folder']=='offline_virtual_k':print(row['job'],row['compiled_constants'],'static',row['static_shared_bytes'],'ops',row.get('opcode_counts'))
print('Audited',len(rows),'cubins')
