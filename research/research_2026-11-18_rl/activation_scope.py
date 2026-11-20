"""Summarize the documented >=512-token dispatch separately from fallback cases."""
import json
import re
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
for split in ("train", "holdout"):
    source = root / f"research_2026-11-18_rl_experts/ballot_{split}/canonical.json"
    data = json.loads(source.read_text())
    assert data["complete"] and data["active"] and data["correct"]
    shapes = data["bench"]["shapes"]
    before = list(shapes)
    for name in before:
        rows = int(re.fullmatch(r"M(\d+)_(uniform|cluster)", name).group(1))
        if rows < 512:
            del shapes[name]
    assert len(shapes) == 6 and len(before) == 14
    data["scope"] += "; subgroup with compiled native activation (M >= 512); 16K is four 4096-row dispatches"
    data["source"] = str(source.relative_to(root))
    (here / f"active_moe_{split}.json").write_text(json.dumps(data, indent=2)+"\n")
