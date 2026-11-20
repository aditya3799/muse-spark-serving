from pathlib import Path
import ast,hashlib
p=Path(__file__).resolve().parent
f=p/'memory_probe.py';s=f.read_text(encoding='utf-8');anchor="if a.mode=='humming':assert not original_ptrs&retained_ptrs"
s=s.replace(anchor,"(a.out/'memory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')\n"+anchor)
ast.parse(s);f.write_bytes(s.encode('utf-8'))
f=p/'offline_sm90_compile.py';s=f.read_text(encoding='utf-8').replace('import argparse,hashlib,json,os,re,sys,time','import argparse,hashlib,json,os,re,shutil,sys,time')
anchor="        cmdline=cubin.parent/'cmdline.json'"
addition="""        copied_cubin=a.out/'kernel.sm90a.cubin'
        shutil.copyfile(cubin,copied_cubin)
        report['cubin_artifact']=str(copied_cubin)
        report['cache_reused']=cubin.stat().st_mtime<started
        for logname in ('stdout.log','stderr.log','cmdline.json'):
            log=cubin.parent/logname
            if log.exists():shutil.copyfile(log,a.out/logname)
"""
assert anchor in s;s=s.replace(anchor,addition+anchor);ast.parse(s);f.write_bytes(s.encode('utf-8'))
print('memory final harness sha',hashlib.sha256((p/'memory_probe.py').read_bytes()).hexdigest())
