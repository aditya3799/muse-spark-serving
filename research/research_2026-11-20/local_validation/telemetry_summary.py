import csv,json,pathlib,statistics
O=pathlib.Path('D:/OpenKernelChallenge/serving/research_2026-11-20/local_validation')
rows=[]
for values in csv.reader((O/'gpu_telemetry.csv').open()):
    if len(values)!=9 or not values[0].strip().startswith('2026/'):continue
    try:
        rows.append(dict(timestamp=values[0].strip(),pstate=values[2].strip(),sm_mhz=float(values[3].split()[0]),memory_mhz=float(values[4].split()[0]),power_w=float(values[5].split()[0]),temperature_c=float(values[6].split()[0]),utilization_gpu_pct=float(values[7].split()[0]),memory_used_mib=float(values[8].split()[0])))
    except ValueError:continue
active=[r for r in rows if r['utilization_gpu_pct']>=90]
doc=dict(samples=len(rows),busy_samples=len(active),first=rows[0]['timestamp'],last=rows[-1]['timestamp'],scope='One-second nvidia-smi samples, not a peak-memory guarantee or DRAM throughput measurement.',dram_bandwidth_measured=False,observed_max_memory_mib=max(r['memory_used_mib'] for r in rows),busy_pstates=sorted({r['pstate'] for r in active}),busy={key:dict(min=min(r[key] for r in active),median=statistics.median(r[key] for r in active),max=max(r[key] for r in active)) for key in ('sm_mhz','memory_mhz','power_w','temperature_c')})
(O/'telemetry_summary.json').write_text(json.dumps(doc,indent=2));print(json.dumps(doc))
