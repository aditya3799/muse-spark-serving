"""Read-only official job monitoring; credentials never written or printed."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

JOBS = ('9689b3114c2a45ab', 'fa0d2a0fcf584339')
BASE = 'https://openkernelchallenge.dev/api/v1'
OUT = Path(__file__).resolve().parent / 'remote_monitor'
OUT.mkdir(exist_ok=True)
stamp = OUT / 'last_poll_epoch.txt'
now = time.time()
if stamp.exists() and now - float(stamp.read_text()) < 110:
    raise SystemExit('Skipped: less than 110 seconds since the previous poll.')
stamp.write_text(str(now), encoding='utf-8')
config = json.loads(Path(r'D:\OpenKernelChallenge\.config\okc\config.json').read_text(encoding='utf-8-sig'))
token = config['token']
del config

def poll(job):
    queried = datetime.now(timezone.utc).isoformat()
    req = Request(BASE + '/jobs/' + job, headers={'Authorization': 'Bearer ' + token,
                  'User-Agent': 'okc-readonly-monitor/1.0', 'Accept': 'application/json'})
    try:
        with urlopen(req, timeout=25) as response:
            data = json.loads(response.read())
            http_status = response.status
    except HTTPError as exc:
        failure = {'job_id': job, 'queried_at': queried, 'http_error': exc.code,
                   'body': exc.read(4096).decode('utf-8', errors='replace').replace(token, '[REDACTED]')}
        (OUT / ('error_' + job + '.json')).write_text(json.dumps(failure, indent=2), encoding='utf-8')
        return failure
    except (URLError, TimeoutError):
        return {'job_id': job, 'queried_at': queried, 'error': 'transport_failure'}
    if data.get('id') != job:
        return {'job_id': job, 'queried_at': queried, 'error': 'unexpected_job_id'}
    previous_path = OUT / ('latest_' + job + '.json')
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else None
    snapshot = {'queried_at': queried, 'endpoint': BASE + '/jobs/' + job,
                'http_status': http_status, 'response': data}
    rendered = json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n'
    with previous_path.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(rendered)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + job + '.json'
    with (OUT / name).open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(rendered)
    return {'job_id': job, 'queried_at': queried, 'status': data.get('status'),
            'previous_status': previous['response'].get('status') if previous else 'queued',
            'transition': data.get('status') != (previous['response'].get('status') if previous else 'queued'),
            'result': data.get('result')}

with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(poll, JOBS))
del token
print(json.dumps(results, indent=2, ensure_ascii=False))
