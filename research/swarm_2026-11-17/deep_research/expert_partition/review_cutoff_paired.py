import ast,collections,hashlib,importlib.util,json,math,random,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
spec=importlib.util.spec_from_file_location('tile_study',ROOT/'optimization_marlin_tiles_2026-11-15/tile_study.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tree=ast.parse((P/'probe_cutoff_paired.py').read_text().replace("pattern.removeprefix('local_count_')","pattern[len('local_count_'):]"));fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='make_routes');ns={'random':random,'stock_make_routes':module.make_routes};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<routes>','exec'),ns)
source=P/'cutoff_paired_01/results.json';data=json.loads(source.read_text());rows=[]
for v in data['fullchain']:
 if 'rank' not in v:continue
 m=v['m'];rank=v['rank'];cutoff=v['plan']['cutoff']//v['cutoff_factor'];global_ids=ns['make_routes'](m,8,256,8,v['pattern'],216+m)
 counts=collections.Counter(e-rank*32 for r in global_ids for e in r if rank*32<=e<(rank+1)*32)
 changed=sum(cutoff<c<=v['plan']['cutoff'] for c in counts.values());stage_work={}
 for stage in ('w13','w2'):
  l=v['plan']['light'][stage]['block_shape'][0];h=v['plan']['heavy'][stage]['block_shape'][0]
  stage_work[stage]={str(f):{'padded_rows':sum(math.ceil(c/(l if c<=cutoff*f else h))*(l if c<=cutoff*f else h) for c in counts.values()),'row_tiles':sum(math.ceil(c/(l if c<=cutoff*f else h)) for c in counts.values())} for f in (1,2)}
 rows.append({**{k:v[k] for k in ('m','rank','pattern','speedup','noise_pass','controls')},'changed_experts':changed,'same_assignment_control':changed==0,'stage_work':stage_work,'A_factor1_us':statistics.median(v['samples_us']['A1']+v['samples_us']['A2']),'B_factor2_us':statistics.median(v['samples_us']['B1']+v['samples_us']['B2'])})
j=json.loads((source.parent/'comparison_journal.json').read_text());report={'scope':'Direct paired local factor1 versus factor2 with unchanged scored-reference numerical gate; two rank labels are not an eight-rank critical-path estimate. All failed controls remain unqualified. No H200 or serving transfer.','status':data['status'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'comparisons':len(j),'all_numerics_bitwise':all(v['bitwise'] for v in j),'all_numerics_pass':all(v['passed'] for v in j),'rows':rows}
(P/'CUTOFF_PAIRED_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':data['status'],'cases':len(rows),'comparisons':len(j),'all_bitwise':report['all_numerics_bitwise'],'qualified':sum(v['noise_pass'] for v in rows),'rank0':[{k:v for k,v in r.items() if k not in ('stage_work','controls')} for r in rows if r['rank']==0]},indent=2))
