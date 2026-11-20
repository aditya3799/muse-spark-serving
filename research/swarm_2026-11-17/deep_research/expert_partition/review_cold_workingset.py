import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;rpath=P/'cold_distinct_layers_01/results.json';r=json.loads(rpath.read_text())
review={'status':r['status'],'scope':r['scope'],'report_sha256':hashlib.sha256(rpath.read_bytes()).hexdigest(),'canonical_total_bytes':r['canonical_total_bytes'],'reported_l2_bytes':r['reported_l2_bytes'],'sweep_to_l2_ratio':r['flush_bytes']/r['reported_l2_bytes'],'cache_scope':r['cache_scope'],'distinct_parameter_pointers':len({v['pointer'] for v in r['layer_storage']})==len(r['layer_storage']),'strict_comparisons':len(r['checks']),'all_bitwise':all(v['bitwise'] for v in r['checks']),'max_peak_allocated':max(c['peak_allocated'] for c in r['cases']),'max_peak_reserved':max(c['peak_reserved'] for c in r['cases']),'rows':[]}
for c in r['cases']:
 for t in c['timings']:
  review['rows'].append({'m':c['m'],'pattern':c['pattern'],'cache_swept':t['cold'],'speedup':t['speedup'],'controls':t['controls'],'noise_pass':t['noise_pass']})
(P/'COLD_WORKINGSET_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'strict_checks':review['strict_comparisons'],'all_bitwise':review['all_bitwise'],'all18timing_controls_pass':all(v['noise_pass'] for v in review['rows'])}))
