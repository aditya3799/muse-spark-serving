"""Save the two November 18 judge checks, then exit."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
JOBS={"mixed_chunk":"b2fa80590214468b","projection":"76866640ab6f4b9f"}

def save(path,text):
    if path.exists() and path.read_text(encoding="utf-8")==text:
        return
    tmp=path.with_suffix(".tmp")
    tmp.write_text(text,encoding="utf-8")
    tmp.replace(path)

def fetch(job):
    command=["wsl.exe","-d","Ubuntu","--exec","env","XDG_CONFIG_HOME=/mnt/d/OpenKernelChallenge/.config",
             "/mnt/d/OpenKernelChallenge/bin/okc","serve","status",job,"--json"]
    r=subprocess.run(command,capture_output=True,timeout=90,encoding="utf-8",errors="replace")
    r.check_returncode()
    data=json.loads(r.stdout)
    if data.get("id")!=job:
        raise ValueError("unexpected job id")
    return data

def collect():
    pending=dict(JOBS)
    while pending:
        for label,job in list(pending.items()):
            try:
                data=fetch(job)
                result=data.get("result") or {}
                save(ROOT/(label+"_judge.json"),json.dumps(data,indent=2,ensure_ascii=False)+"\n")
                lines=[f"Job {job} | {data.get('status')}"]
                if result:
                    lines+=["Correctness: "+str(result.get("correctness",{}).get("passed")),
                            "Overall pass: "+str(result.get("passed")),
                            "Later turns: "+str(result.get("later_turns")),
                            "Over-stock memory: "+str(result.get("gpu_memory")),
                            "Relative to judge stock: "+str(result.get("vs_stock")),
                            "Rungs: "+json.dumps(result.get("rungs"),ensure_ascii=False),
                            "Projection activation line observed in returned startup log: "+str("Sharding DFlash context projection" in str(result.get("startup_log","")))]
                save(ROOT/(label+"_status.txt"),"\n".join(lines)+"\n")
                if data.get("finished_at") or data.get("status") in {"done","failed","canceled","cancelled","rejected","timed_out","error"}:
                    del pending[label]
            except (OSError, ValueError,subprocess.SubprocessError) as exc:
                print(datetime.now(timezone.utc),label,type(exc).__name__,str(exc),flush=True)
        if pending:
            time.sleep(60)
if __name__=="__main__":
    collect()
