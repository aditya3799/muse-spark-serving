import pathlib,hashlib,json,datetime
p=pathlib.Path('serving/research_2026-11-19_round2/moe');f=p/'ROUND2_MOE_EVIDENCE.json';r=json.loads(f.read_text());e=p/'PR50800_FROZEN_SCOPE_ERRATA.txt';r['updated_utc']=datetime.datetime.utcnow().isoformat()+'Z';r['evidence'].append(dict(path=e.name,sha256=hashlib.sha256(e.read_bytes()).hexdigest(),scope='Independent raw-scale source review and inherited-comment erratum',status='Frozen source unchanged; finiteCPU/math versus byte-layout scopes clarified'))
f.write_text(json.dumps(r,indent=2));print(hashlib.sha256(f.read_bytes()).hexdigest())
