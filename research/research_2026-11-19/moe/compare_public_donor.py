from pathlib import Path
import ast,re,subprocess,hashlib,json,difflib
here=Path(__file__).resolve().parent;root=here.parents[1]
def source(p):return next(ast.literal_eval(x.value) for x in ast.parse(p.read_text(encoding='utf8')).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in x.targets))
flags=['-D__CUDA_ARCH__=900','-DOKC_NW=16','-DOKC_FAST1=0','-DOKC_EPI4=0','-DOKC_WGS=1','-DOKC_KPERM3=1','-DOKC_PRMTSC=1','-DOKC_NS=3','-DOKC_NSF=3','-DOKC_FNST=4']
outputs=[]
for name,p in [('existing',root.parent/'muse_moe_w4a16/submission_v139_wgs_kperm3.py'),('public',here/'alygg77/muse_moe_w4a16.py')]:
 src=re.sub(r'^\s*#include[^\n]*','',source(p),flags=re.M)
 rr=subprocess.run(['g++','-E','-P','-x','c++',*flags,'-'],input=src,capture_output=True,text=True)
 assert rr.returncode==0,rr.stderr
 outputs.append(rr.stdout)
 (here/'alygg77'/(name+'_preprocessed.cu')).write_text(rr.stdout)
def body(src,name):
 matches=list(re.finditer(r'\b'+name+r'\s*\([^;{}]*\)\s*\{',src))
 if not matches:return None
 a=matches[0].end()-1;depth=1;b=a+1
 while depth:
  depth+=(src[b]=='{')-(src[b]=='}');b+=1
 return re.sub(r'\s+','',src[a:b])
rows=[]
for name in ['wgs_seg','wgs_epi_tile','wgs_flush','wgs_prefix','wgs_copy','route','moe_fused','run_phase','wgs_stream']:
 a,b=(body(x,name) for x in outputs)
 rows.append(dict(function=name,existing_length=len(a) if a else None,public_length=len(b) if b else None,identical=a==b and a is not None))
(here/'alygg77/preprocessed_comparison.json').write_text(json.dumps(dict(flags=flags,functions=rows),indent=2));print(json.dumps(rows,indent=2))
