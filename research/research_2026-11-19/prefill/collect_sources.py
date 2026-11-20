"""Bounded primary-source snapshots for the November 19 prefill investigation."""
import datetime
import hashlib
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "sources"
DEST.mkdir(exist_ok=True)
MAX_BYTES = 2_000_000

SOURCES = {
    "trt_release": "https://api.github.com/repos/NVIDIA/TensorRT-LLM/releases/tags/v1.4.0rc4",
    "trt_muse_pr": "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/21049",
    "trt_muse_files": "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/21049/files?per_page=100",
    "trt_prompt_pr": "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/21308",
    "trt_prompt_files": "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/21308/files?per_page=100",
    "trt_host_pr": "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/20848",
    "trt_decode_files": "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/21356/files?per_page=100",
    "trt_dwdp_pr": "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/21012",
    "trt_ar_pr": "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/20241",
    "lightllm_commit": "https://api.github.com/repos/ModelTC/LightLLM/commits/main",
    "fastdeploy_release": "https://api.github.com/repos/PaddlePaddle/FastDeploy/releases/latest",
    "lmdeploy_release": "https://api.github.com/repos/InternLM/lmdeploy/releases/latest",
    "dynamo_release": "https://api.github.com/repos/ai-dynamo/dynamo/releases/latest",
    "ssengine_commit": "https://api.github.com/repos/SyFI-ServingStudio/SSEngine-GLM5.3-Flash/commits/main",
}

def main():
    records = []
    for key, url in SOURCES.items():
        path = DEST / (key + ".json")
        record = {"key": key, "url": url, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "okc-prefill-research", "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(request, timeout=25) as response:
                data = response.read(MAX_BYTES + 1)
                record["status"] = response.status
            if len(data) > MAX_BYTES:
                raise ValueError("per-file byte budget exceeded")
            parsed = json.loads(data)
            path.write_bytes(data)
            record.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), path=str(path.relative_to(HERE)))
            if isinstance(parsed, dict):
                record.update({k: parsed.get(k) for k in ("sha", "tag_name", "published_at", "merge_commit_sha", "merged_at", "title")})
        except Exception as exc:
            record["error"] = repr(exc)
        records.append(record)
        print(key, record.get("status"), record.get("bytes"), record.get("error", ""), flush=True)
        (HERE / "source_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        time.sleep(.25)

if __name__ == "__main__":
    main()
