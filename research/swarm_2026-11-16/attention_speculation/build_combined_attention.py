"""Combine frozen source exports, never patch active SGLang or root packages."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
OUT = BASE / "combined_attention_optional_v1"
FA = "python/sglang/srt/layers/attention/flashattention_backend.py"
META = "python/sglang/kernels/ops/attention/metadata.py"
INPUTS = {
    "scheduler": (BASE / "candidate" / FA, "2a720c8f5ed86437e6ec181cace03ae97fba0460672b1d001549f8df4b71e03c"),
    "draft_glue": (BASE / "draft_glue/candidate" / FA, "b108812fb264933237b0e92accbe943b5bd5ae3b5ae12d5d5dc3f341f3afaae1"),
    "prefix_scan": (BASE / "prefix_scan/candidate" / META, "300cb9428d6281c10812be0a302e88d82dff9f81baf5cab1087730fe8bc6df3c"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    text = data.decode("utf-8").replace("\r\n", "\n")
    assert "\r" not in text and not text.startswith("\ufeff")
    return text


def main():
    sources, provenance = {}, {}
    for label, (path, expected) in INPUTS.items():
        raw = path.read_bytes()
        assert digest(raw) == expected, (label, "frozen source changed")
        sources[label] = canonical(raw)
        provenance[label] = dict(path=str(path), sha256=expected)
    glue_tree = ast.parse(sources["draft_glue"])
    glue_cls = next(n for n in glue_tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashAttentionBackend")
    glue = next(n for n in glue_cls.body if isinstance(n, ast.FunctionDef) and n.name == "glue_draft_metadata")
    glue_text = "".join(sources["draft_glue"].splitlines(True)[glue.lineno - 1:glue.end_lineno]) + "\n\n"
    needle = "    def validate_elastic_cuda_graph_recapture(self) -> None:"
    assert sources["scheduler"].count(needle) == 1
    assert "    def glue_draft_metadata(" not in sources["scheduler"]
    combined = sources["scheduler"].replace(needle, glue_text + needle)
    tree = ast.parse(combined)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashAttentionBackend")
    inserted = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "glue_draft_metadata"]
    assert len(inserted) == 1 and ast.dump(inserted[0]) == ast.dump(glue)
    cls.body.remove(inserted[0])
    assert ast.dump(tree) == ast.dump(ast.parse(sources["scheduler"])), "scheduler AST lost or changed"
    outputs = {FA: combined, META: sources["prefix_scan"]}
    manifest = dict(status="OPTIONAL_UNSUBMITTED_CPU_ONLY_COMBINATION", encoding="UTF-8 LF without BOM",
        provenance=provenance, files={}, conflicts=[],
        combination="Exact frozen scheduler plus exact narrow glue method in same class; prefix scan copied byte-for-byte as separate dependency file",
        validation_gaps=["No combined GPU execution, CUDA graph replay or timing", "No full draft attention or acceptance validation",
            "Prefix scan lacks Triton compilation and XPU validation", "Prefix scan arbitrary cross-tensor aliasing not covered",
            "No full-model FP8 H200 correctness, memory or throughput result"])
    delta = []
    for rel, text in outputs.items():
        raw = text.encode("utf-8")
        assert max(map(len, text.splitlines())) <= 400
        path = OUT / "candidate" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        original_raw = (ROOT / "sglang" / rel).read_bytes()
        original = canonical(original_raw)
        manifest["files"][rel] = dict(sha256=digest(raw), original_sha256=digest(original_raw),
            original_normalized_sha256=digest(original.encode("utf-8")))
        delta += difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile="a/" + rel, tofile="b/" + rel)
    patch = "".join(delta).encode("utf-8")
    (OUT / "combined_delta.diff").write_bytes(patch)
    manifest["diff"] = dict(sha256=digest(patch), bytes=len(patch))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
