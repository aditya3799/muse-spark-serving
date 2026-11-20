"""Filter public agent trajectories to Muse Spark Open at the recorded revision."""
import hashlib
import json
from pathlib import Path
import requests

HERE=Path(__file__).resolve().parent
REV="8cfdcf2d8384d0d765eca390e5bfba067e3b2899"
BASE=f"https://huggingface.co/datasets/PatronusAI/few-word-do-trick-traces/resolve/{REV}/data/"
selected=[]
files=[]
for name in ("swe-bench-pro","swe-bench-verified","terminal-bench-2.1"):
    with requests.get(BASE+name+".jsonl",stream=True,timeout=(20,90)) as response:
        response.raise_for_status()
        seen=0
        count=0
        for line in response.iter_lines(chunk_size=524288):
            if not line: continue
            row=json.loads(line)
            seen+=1
            model=str(row.get("model",""))+" "+str(row.get("model_id",""))+" "+str(row.get("model_name",""))
            if "muse" in model.lower() and "spark" in model.lower():
                selected.append(row)
                count+=1
        files.append(dict(file=name,entries=seen,muse=count))
assert len(selected)==20,files
out=HERE/"patronus_muse.jsonl"
out.write_bytes(b"".join(json.dumps(r,ensure_ascii=False).encode("utf-8")+b"\n" for r in selected))
print(files,hashlib.sha256(out.read_bytes()).hexdigest())
print("fields",list(selected[0]),"step fields",[(step.get("source"),list(step)) for step in selected[0]["steps"][:4]])

