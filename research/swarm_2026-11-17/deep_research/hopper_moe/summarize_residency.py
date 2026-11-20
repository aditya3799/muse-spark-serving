from pathlib import Path
import json,re,hashlib
P=Path(__file__).resolve().parent;OLD=P.parents[2]/'swarm_2026-11-16/moe/phase4_hour'
rows=[]
for folder in ('offline_occupancy_m8','offline_occupancy_cpasync'):
 path=P/folder/'results.json';data=json.loads(path.read_text())
 for r in data['cases']:
  lines=' '.join(r['resource_lines']);regs=int(re.search(r'Used (\d+) registers',lines).group(1));threads=r['resolved']['total_threads'];smem=r['shared_bytes_estimated'];rounded=((regs+7)//8)*8
  rows.append({'job':r['job_id'],'registers_per_thread':regs,'threads':threads,'shared_bytes_estimated':smem,'register_model_cta_bound':65536//(threads*rounded),'shared_model_cta_bound':(228*1024)//smem,'thread_model_cta_bound':2048//threads,'launch_ctas_per_sm':r['configuration']['num_ctas_per_sm'],'spills':lines,'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
coverage=[]
for folder in ('producer_all_00','producer_all_22','producer_all_44','producer_all_66'):
 path=OLD/folder/'results.json';data=json.loads(path.read_text())
 for r in data['cases']:
  if r['policy']!='tma_b_only':continue
  c=r['configuration'];shared=r['shared_bytes_estimated'];threads=r['resolved']['total_threads']
  coverage.append({'stage':r['stage'],'block_shape':c['block_shape'],'warp_shape':c['warp_shape'],'intervals':r['intervals'],'shared_bytes_estimated':shared,'threads':threads,'two_ctas_fit_shared':shared*2<=228*1024,'resource_lines':r['resource_lines']})
report={'status':'static_resource_comparison','warning':'Resource bounds are necessary capacity checks, not runtime occupancy. Hardware residency requires CUDA occupancy API or observed block occupancy; compiler launchbounds alone neither guarantees concurrency nor gain.','rows':rows,'upper_interval_existing_compile_evidence':coverage}
(P/'RESIDENCY_RESOURCE_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print('new matched resource rows',len(rows),'unique upper geometry rows',len(coverage))
for r in coverage:
 if r['block_shape'][0]!=8:print(r['stage'],r['block_shape'],'shared',r['shared_bytes_estimated'],'2CTAshared',r['two_ctas_fit_shared'])
