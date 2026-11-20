"""Chronological reconstruction from stored ATIF fields, using actual provider text.

The original provider chat-template and token stream are unavailable. The split
view treats field boundaries conservatively; the joined view inserts newlines.
Only proposal matching is replayed. Neither view includes actual DFlash logits.
"""
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def samples(joined=False):
    for line in (HERE/"patronus_muse.jsonl").read_bytes().splitlines():
        row=json.loads(line)
        assert row["model_id"]=="openrouter/meta/muse-spark-open"
        messages=[]
        for step in row["steps"]:
            if step["source"]=="user":
                messages.append(dict(role="user",content=step["message"]))
                continue
            assert step["source"]=="agent"
            pieces=[step.get("reasoning_content"),step.get("message")]
            for call in step.get("tool_calls",[]):
                key=call.get("arguments",{}).get("keystrokes")
                if isinstance(key,str):
                    pieces.append(key)
            pieces=[s for s in pieces if s]
            if joined:
                pieces=["\n".join(pieces)]
            messages.extend(dict(role="assistant",content=s) for s in pieces)
            # The environment responses enter history after the generated text.
            messages.extend(dict(role="tool",content=obs["content"])
                            for obs in step["observation"]["results"] if obs["content"])
        yield messages


if __name__=="__main__":
    rows=[json.loads(line) for line in (HERE/"patronus_muse.jsonl").read_bytes().splitlines()]
    met=[s["metrics"] for r in rows for s in r["steps"] if s["source"]=="agent"]
    print("traces",len(rows),"agent turns",len(met),"recorded completion tokens",sum(m["completion_tokens"] for m in met),
          "median recorded prompt",statistics.median(m["prompt_tokens"] for m in met),
          "maximum recorded prompt",max(m["prompt_tokens"] for m in met))

