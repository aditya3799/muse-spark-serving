import sys,subprocess,resource,pathlib
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
try:pathlib.Path('/proc/self/coredump_filter').write_text('0')
except OSError:pass
here=pathlib.Path(__file__).resolve().parent
for rel in ('gather_norm/build_production_stream.py','finalize_reduce/build.py','finalize_reduce/probe.py'):
 print('RUN',rel,flush=True)
 r=subprocess.run([sys.executable,str(here/rel)],timeout=180)
 print('EXIT',rel,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
