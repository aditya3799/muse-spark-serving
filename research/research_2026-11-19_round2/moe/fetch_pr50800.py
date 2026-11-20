import json,pathlib,urllib.request,hashlib,concurrent.futures
p=pathlib.Path('serving/research_2026-11-19_round2/moe/sources/nov10_sweep'); m=json.loads((p/'sglang_pr_manifest.json').read_text()); pr=next(x for x in m['prs'] if x['number']==50800)
files=[x for x in pr['files'] if 'nvfp4' in x['filename']]; dest=p/'pr50800'; dest.mkdir(exist_ok=True)
def get(f):
 u='https://raw.githubusercontent.com/sgl-project/sglang/'+pr['head']+'/'+f['filename']; b=urllib.request.urlopen(u,timeout=30).read(); out=dest/pathlib.Path(f['filename']).name;out.write_bytes(b);return dict(path=str(out),upstream_path=f['filename'],url=u,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:r=list(ex.map(get,files))
(dest/'manifest.json').write_text(json.dumps({'head':pr['head'],'files':r},indent=2));print(json.dumps(r,indent=2))
