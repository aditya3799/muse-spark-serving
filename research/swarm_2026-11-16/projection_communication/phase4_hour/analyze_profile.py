import argparse,json,statistics,hashlib
from pathlib import Path
pa=argparse.ArgumentParser();pa.add_argument('run',type=Path);a=pa.parse_args();d=json.loads((a.run/'results.json').read_text());rows=[]
for r in d['rows']:
 if 'event_graph_stages' not in r:continue
 stages=r['event_graph_stages'];keys=stages[0]['stages'];times={k:statistics.median(x['stages'][k]['stream_us'] for x in stages) for k in keys};total=sum(times.values())
 e=r.get('requested_distinct_local_experts');suffix='' if e is None else '_e%d'%e
 trace=a.run/('graph_m%d%s.trace.json'%(r['m'],suffix));events=json.loads(trace.read_text())['traceEvents'];kernels=[x for x in events if x.get('cat')=='kernel']
 kernel_sum=sum(x['dur'] for x in kernels);gemm_sum=sum(x['dur'] for x in kernels if 'humming<' in x['name'])
 plan=r['selected_split_plan'];counts=r['expert_counts'];tiles={}
 if plan:
  for stage in ('w13','w2'):
   tiles[stage]={}
   for partition in ('light','heavy'):
    bm=plan[partition][stage]['block_shape'][0]
    chosen=[c for c in counts if c>0 and ((c<=plan['cutoff'])==(partition=='light'))]
    tiles[stage][partition]={'BM':bm,'experts':len(chosen),'active_rows':sum(chosen),'padded_row_tiles':sum((c+bm-1)//bm for c in chosen),'padded_rows':sum((c+bm-1)//bm*bm for c in chosen)}
 rows.append({'m':r['m'],'pattern':r['pattern'],'active_routes':r['active_routes'],'distinct_local_experts':sum(c>0 for c in counts),'counts':counts,'stage_graph_median_us':times,'stage_graph_fraction':{k:v/total for k,v in times.items()},'graph_stream_us':r['plain_graph_before']['median_stream_interval_us'],'eager_stream_us':r['eager_before']['median_stream_interval_us'],'eager_host_call_us':r['eager_before']['median_host_call_us'],'graph_host_call_us':r['plain_graph_before']['median_host_call_us'],'controls':r['same_arm_controls'],'noise_pass':r['same_arm_noise_pass'],'instrumentation_ratio':r['graph_instrumentation_ratio'],'changed_input_exact':r['changed_input_graph_exact'],'kernel_trace':{'count':len(kernels),'sum_us':kernel_sum,'humming_us':gemm_sum,'humming_fraction':gemm_sum/kernel_sum,'scope':'Actual trace category kernel only, excludes duplicated profiler CPU/device aggregate events'},'padding_from_logged_histogram_and_config':tiles})
summary={'status':'ANALYZED','run':str(a.run),'source_report_sha256':hashlib.sha256((a.run/'results.json').read_bytes()).hexdigest(),'rows':rows,'scope':'Local source-exact diagnostic only. Stage event medians and separate CUPTI traces corroborate ordering, but neither estimates H200 serving share. Total instrumentation overhead does not guarantee no per-stage bias.','memory_scope':'Reported peak includes weight preparation; runtime-only graph peak not isolated in this profiler.'}
(a.run/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'rows':len(rows),'humming_kernel_fractions':[r['kernel_trace']['humming_fraction'] for r in rows]}))
