import pathlib,sys,resource,json,importlib.util,hashlib,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
HERE=pathlib.Path(__file__).resolve().parent
path=pathlib.Path(sys.argv[1]);out=HERE/((path.parent.name+"_" if path.name=="sass.txt" else "")+path.stem+"_audit.json")
spec=importlib.util.spec_from_file_location("hazard",HERE/"python__sglang__kernels__jit__utils__wgmma_rs_hazard.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
text=path.read_text();start=time.monotonic();report=dict(source_sass=str(path),sass_sha256=hashlib.sha256(text.encode()).hexdigest(),checker_sha256=hashlib.sha256(pathlib.Path(spec.origin).read_bytes()).hexdigest(),complete=False,functions=[])
def save():out.write_text(json.dumps(report,indent=2)+"\n")
save()
try:
 functions=mod.parse_sass(text)
 if not functions:raise RuntimeError("No parsedfunctions; cannot claimpassed")
 for name,ins in functions.items():
  total=sum(bool(x.gmma) for x in ins)
  if not total:continue
  rs,issues,complete=mod.check_function(ins)
  item=dict(name=name,instructions=len(ins),gmma=total,rs_gmma=rs,analysis_complete=complete,problem_count=len(issues),first_problems=issues[:16])
  report['functions'].append(item);save();print(json.dumps({k:v for k,v in item.items() if k!='first_problems'}),flush=True)
 report.update(complete=True,elapsed_seconds=time.monotonic()-start,rs_gmma=sum(x['rs_gmma'] for x in report['functions']),all_analyses_complete=all(x['analysis_complete'] for x in report['functions']),problem_count=sum(x['problem_count'] for x in report['functions']))
except Exception as e:report['error']=type(e).__name__+': '+str(e)
save();print(json.dumps({k:v for k,v in report.items() if k!='functions'}),flush=True)
