import urllib.request,json,pathlib,hashlib
p=pathlib.Path('serving/research_2026-11-19_round2/moe/sources/nov10_sweep/other_engines');base='https://api.github.com/repos/InternLM/lmdeploy/pulls/5173';r={}
for name,url in [('metadata',base),('files',base+'/files?per_page=100')]:
 b=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'okc-research'}),timeout=30).read();(p/('lmdeploy_5173_'+name+'.json')).write_bytes(b);d=json.loads(b);r[name+'_sha256']=hashlib.sha256(b).hexdigest()
 if name=='metadata':r.update(head=d['head']['sha'],created=d['created_at'],merged=d['merged'],title=d['title'],body=d['body'])
 else:r['files']=[dict(path=x['filename'],additions=x['additions']) for x in d]
(p/'lmdeploy_5173_manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
