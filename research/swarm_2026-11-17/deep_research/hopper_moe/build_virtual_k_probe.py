from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'offline_occupancy_sm90.py').read_text().replace("P/'occupancy_jobs.json'","P/'virtual_k_jobs.json'").replace('default=12','default=5')
s=s.replace('from producer_tuning import producer_config','from producer_tuning import producer_config\nfrom virtual_k_codegen import render_virtual_k')
s=s.replace("files=[OLD/'producer_tuning.py'","files=[P/'virtual_k_codegen.py',*sorted(P.glob('virtual_k_*.cuh')),OLD/'producer_tuning.py'")
s=s.replace("row['shared_bytes_estimated']=k.estimated_smem_size", """generated,virtual_manifest=render_virtual_k(k.code,configuration,90,enabled=policy=='virtual')
  row['virtual_manifest']=virtual_manifest
  row['shared_bytes_estimated']=k.estimated_smem_size""")
s=s.replace('(dest/\'kernel.cu\').write_text(k.code)','(dest/\'kernel.cu\').write_text(generated)').replace('hashlib.sha256(k.code.encode())','hashlib.sha256(generated.encode())').replace('Compiler.compile(k.code,','Compiler.compile(generated,')
s=s.replace("'total_threads':k.num_threads", "'total_threads':virtual_manifest.get('physical_threads',k.num_threads),'reference_threads':k.num_threads")
s=s.replace("'math_threads':k.num_math_threads", "'math_threads':virtual_manifest.get('physical_threads',k.num_math_threads)").replace("'load_threads':k.num_load_threads", "'load_threads':virtual_manifest.get('physical_threads',k.num_load_threads)")
(P/'offline_virtual_k_sm90.py').write_text(s)
print('Virtual-K offline harness prepared, no compilation')
