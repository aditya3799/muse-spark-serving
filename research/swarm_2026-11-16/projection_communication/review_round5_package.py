from pathlib import Path
import ast,hashlib,json,re,subprocess
b=Path(r'D:\OpenKernelChallenge\serving');sw=b/'swarm_2026-11-16';p=sw/'coordination/round5_humming_adaptive_fused';old=sw/'coordination/round4_humming_masked';own=sw/'projection_communication';sha=lambda x:hashlib.sha256(x).hexdigest()
validation=json.loads((p/'package_validation.json').read_text());prior=json.loads((old/'package_validation.json').read_text());manifest=json.loads((own/'adaptive_split_v3/manifest.json').read_text());patch=(p/'candidate.diff').read_bytes();assert sha(patch)==validation['patch_sha256'];assert len(patch)==validation['patch_bytes'];text=patch.decode('utf-8');assert '\r' not in text
headers=list(re.finditer(r'^diff --git a/(.+) b/(.+)\n',text,re.M));assert len(headers)==validation['file_count']==10
rows=[]
for i,h in enumerate(headers):
 name=h.group(1);assert name==h.group(2) and name.startswith('python/sglang/') and '..' not in Path(name).parts
 section=text[h.end():headers[i+1].start() if i+1<len(headers) else len(text)];assert not any(x in section for x in ('GIT binary patch','rename from ','deleted file mode','old mode ','new mode '))
 new='new file mode 100644\n' in section
 modes=re.findall(r'^(?:new file mode |index [^\n]+ )([0-9]{6})$',section,re.M);assert modes and all(m=='100644' for m in modes)
 if new:before=''
 else:
  q=subprocess.run(['git','-C',str(b/'sglang'),'show',validation['base']+':'+name],capture_output=True,check=True);before=q.stdout.decode('utf-8')
 source=before.splitlines(True);result=[];pos=0;lines=section.splitlines(True);j=0
 while j<len(lines):
  match=re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',lines[j])
  if not match:j+=1;continue
  start=max(int(match.group(1))-1,0);oldcount=int(match.group(2) or 1);newcount=int(match.group(4) or 1)
  assert start>=pos;result.extend(source[pos:start]);pos=start;j+=1;a=c=0
  while j<len(lines) and not lines[j].startswith('@@ '):
   line=lines[j]
   if line.startswith(' '):assert source[pos]==line[1:];result.append(line[1:]);pos+=1;a+=1;c+=1
   elif line.startswith('-'):assert source[pos]==line[1:];pos+=1;a+=1
   elif line.startswith('+'):result.append(line[1:]);c+=1
   elif line.startswith('\\ No newline'):raise AssertionError('Unexpected newline marker')
   else:raise AssertionError(('Unexpected patch line',line))
   j+=1
  assert a==oldcount and c==newcount,(name,a,oldcount,c,newcount)
 result.extend(source[pos:]);rebuilt=''.join(result).encode('utf-8');actual=(p/'candidate_sources'/name).read_bytes();assert rebuilt==actual;assert sha(actual)==validation['source_hashes'][name];ast.parse(actual.decode('utf-8'))
 ownmatch=None
 if name in manifest['sources']:ownmatch=sha(actual)==manifest['sources'][name]==sha((own/'adaptive_split_v3/serving_export'/name).read_bytes());assert ownmatch
 retained=name in prior['source_hashes'] and prior['source_hashes'][name]==sha(actual)
 rows.append({'path':name,'sha256':sha(actual),'regular_mode':'100644','new_file':new,'independent_base_hunk_rebuild_exact':True,'own_frozen_export_match':ownmatch,'unchanged_from_round4':retained})
assert {x['path'] for x in rows if not x['unchanged_from_round4']}=={'python/sglang/kernels/ops/moe/muse_spark_humming_split.py','python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py','python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py','python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'}
report={'status':'PACKAGE_PASS_GPU_AND_HOPPER_GATES_PENDING','patch_sha256':sha(patch),'patch_bytes':len(patch),'base':validation['base'],'files':rows,'intended_delta_from_round4':['New adaptive split module','Parent forwards true global expert count and v3 recognized-norm scale deferral','NVFP4 default helper retained plus split wrapper','v3 BF16 scale rounding fused in finalizer'],'retained_source_count':sum(x['unchanged_from_round4'] for x in rows),'review_scope':'Independent read-only git object reads and unified hunk replay in memory; no checkout/index/source modifications; no GPU/compiler launch','gpu_reports':'pending','remaining_h200_limits':['Actual SM90 numerical order and end-to-end output-token checks','Full-image imports/backend startup and model loader','Multi-GPU projection/collective CUDA graphs','Whole-model peak memory and capture buckets','Serving throughput/TTFT including adaptive extra launches; no1.6x demonstrated'],'required_runtime_flag':'--moe-runner-backend humming'}
(own/'ROUND5_FINAL_GATE_REVIEW.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps({'status':report['status'],'files':len(rows),'retained':report['retained_source_count'],'patch_sha256':report['patch_sha256']}))
