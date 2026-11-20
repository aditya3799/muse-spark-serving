"""Recover only two previously hashed public tensor payloads by HTTP byte range.

Preserves historical receipts and refuses a full-file HTTP response. Total
recovered payload is below700MiB; no complete checkpoint is downloaded.
"""
import hashlib,json,shutil,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'research_2026-11-18_rl';sys.path.insert(0,str(OLD))
from public_fc import partial

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(16<<20),b''):h.update(data)
    return h.hexdigest()

def main():
    total=0;started=time.time()
    for kind,extension in [('fc','bf16'),('aux','raw')]:
        original=json.loads((OLD/f'public_{kind}_receipt.json').read_text())
        destination=Path(tempfile.gettempdir())/f'okc-20261119-dflash-{kind}-46c90f412549.{extension}'
        if destination.exists() and destination.stat().st_size==original['bytes']:
            assert digest(destination)==original['sha256']
        else:
            previous=destination.stat().st_size if destination.exists() else 0
            assert previous<=original['bytes']
            assert shutil.disk_usage(destination.parent).free>original['bytes']-previous+(512<<20)
            if kind=='fc':ranges=[(original['first_byte'],original['last_byte'],0)]
            else:ranges=[(r['first_byte'],r['last_byte'],r['offset']) for r in original['tensors'].values()]
            with destination.open('ab') as stream:
                for first,last,offset in ranges:
                    for start in range(first,last+1,16<<20):
                        end=min(start+(16<<20)-1,last)
                        output_start=offset+start-first
                        if output_start+(end-start+1)<=previous:continue
                        assert output_start==stream.tell(),(output_start,stream.tell())
                        for attempt in range(3):
                            try:data=partial(start,end);break
                            except (OSError,TimeoutError):
                                if attempt==2:raise
                        stream.write(data);stream.flush();total+=len(data)
                        print(kind,stream.tell(),'/',original['bytes'],flush=True)
            assert destination.stat().st_size==original['bytes']
            assert digest(destination)==original['sha256']
        report=dict(original,temporary_file=str(destination.resolve()),recovered_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (HERE/f'recovered_public_{kind}_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        print('verified',kind,destination,original['sha256'],flush=True)
    (HERE/'recovery_summary.json').write_text(json.dumps(dict(downloaded_bytes=total,elapsed_seconds=time.time()-started),indent=2)+'\n')

if __name__=='__main__':main()
