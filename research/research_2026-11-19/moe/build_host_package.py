from pathlib import Path
import importlib.util,hashlib,json,resource,time,os,sys
here=Path(__file__).resolve().parent
os.environ['CUDA_HOME']=str(Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13')
os.environ['CUDA_VISIBLE_DEVICES']=''
spec=importlib.util.spec_from_file_location('packaged_group_builder',here/'normal_package/muse_spark_grouped_build.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b._headers=lambda:here/'cutlass_judge/include'
start=time.monotonic()
try:
 path=b.library_path()
 data=path.read_bytes()
 (here/'libgroup_normal_t64_sm90a.so').write_bytes(data)
 report=dict(ok=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),local_header_resolution_override_only=True,cache_path=str(path))
except Exception as exc:
 report=dict(ok=False,error=str(exc),local_header_resolution_override_only=True)
report.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
(here/'normal_host_package.json').write_text(json.dumps(report,indent=2));print(report)
raise SystemExit(0 if report['ok'] else 1)
