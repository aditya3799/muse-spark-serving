"""Persist the judge receipt and terminal result for the isolated Hopper serving check."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"research_2026-11-18_dflash_projection"))
from collect_checks import fetch,save

JOB="0afdc0b191dd4b42"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--once",action="store_true")
    once=parser.parse_args().once
    while True:
        try:
            data=fetch(JOB)
            save(HERE/"native_package_02/h200_job.json",json.dumps(data,indent=2,ensure_ascii=False)+"\n")
            state=data.get("status")
            if data.get("finished_at") or state in {"done","failed","canceled","cancelled","rejected","timed_out","error","passed"}:
                return
        except (OSError,ValueError,RuntimeError) as error:
            if once: raise
            print(datetime.now(timezone.utc),repr(error),file=sys.stderr,flush=True)
        except Exception as error:
            if once: raise
            print(datetime.now(timezone.utc),repr(error),file=sys.stderr,flush=True)
        if once: return
        time.sleep(180)

if __name__=="__main__":
    main()
