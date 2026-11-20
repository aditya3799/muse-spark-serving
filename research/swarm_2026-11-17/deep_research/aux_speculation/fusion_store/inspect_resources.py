from pathlib import Path
import importlib.metadata as metadata,json,subprocess
P=Path(__file__).resolve().parent
bins=[]
for dist in metadata.distributions():
    if 'nvidia' in dist.metadata['Name']:
        bins.extend(str(dist.locate_file(f)) for f in dist.files if '/bin/' in str(f) and str(f).endswith('cuobjdump'))
libs=list(Path.home().glob('.cache/sglang/jit/sm86/*norm_residual*/*/*.so'))
report={'cuobjdump':bins,'libraries':[str(p) for p in libs],'resource_reports':[]}
if bins:
    for lib in libs:
        r=subprocess.run([bins[0],'--dump-resource-usage',str(lib)],capture_output=True,text=True,timeout=20)
        report['resource_reports'].append({'library':str(lib),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
(P/'resource_inspection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'tools':bins,'libraries':len(libs),'reports':len(report['resource_reports'])}))
