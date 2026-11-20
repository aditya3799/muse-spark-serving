from pathlib import Path
import hashlib,json,urllib.request,zipfile,os,stat,time
P=Path(__file__).resolve().parent
out=P/'sanitizer_tool';out.mkdir(exist_ok=True)
endpoint='https://pypi.org/pypi/nvidia-cuda-sanitizer-api/13.5.25/json'
with urllib.request.urlopen(endpoint,timeout=30) as response:data=json.load(response)
choices=[r for r in data['urls'] if 'manylinux_2_25_x86_64.whl' in r['filename']]
assert len(choices)==1,choices
entry=choices[0]
assert entry['url'].startswith('https://files.pythonhosted.org/')
wheel=out/entry['filename']
if not wheel.exists():
 with urllib.request.urlopen(entry['url'],timeout=60) as response:wheel.write_bytes(response.read())
digest=hashlib.sha256(wheel.read_bytes()).hexdigest()
assert digest==entry['digests']['sha256']
with zipfile.ZipFile(wheel) as archive:
 members=archive.infolist()
 executables=[r.filename for r in members if r.filename.endswith('/compute-sanitizer')]
 assert executables, 'Wheel has no compute-sanitizer executable'
 target=out/'extracted';target.mkdir(exist_ok=True);root=target.resolve()
 for member in members:
  dest=(target/member.filename).resolve()
  assert dest==root or root in dest.parents,'Unsafe wheel member'
  mode=member.external_attr>>16
  assert not stat.S_ISLNK(mode),'Unexpected wheel symlink'
 archive.extractall(target)
 for member in members:
  dest=target/member.filename
  if dest.is_file() and member.external_attr>>16:os.chmod(dest,(member.external_attr>>16)&0o777)
 for exe in executables:os.chmod(target/exe,0o755)
manifest={'downloaded_utc_seconds':time.time(),'metadata_endpoint':endpoint,'name':data['info']['name'],'version':data['info']['version'],'author':data['info'].get('author'),'maintainer':data['info'].get('maintainer'),'homepage':data['info'].get('home_page'),'project_urls':data['info'].get('project_urls'),'wheel':entry['filename'],'wheel_url':entry['url'],'sha256':digest,'matches_pypi_digest':True,'executables':[str(target/e) for e in executables],'executables_sha256':{e:hashlib.sha256((target/e).read_bytes()).hexdigest() for e in executables},'no_environment_or_driver_install':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
