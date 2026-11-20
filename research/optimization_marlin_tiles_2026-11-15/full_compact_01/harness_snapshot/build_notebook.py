"""Package a self-contained structural notebook for this new experiment phase."""

import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "tile_study.py").read_text(encoding="utf-8")
cells = []


def cell(kind, text):
    result = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == "code":
        ast.parse(text)
        result.update(execution_count=None, outputs=[])
    cells.append(result)


cell(
    "markdown",
    """# Marlin EP tile selection — experiment 01

New phase after the completed mapping study. This notebook measures **routing geometry**, not Marlin latency.
CPU-only is sufficient for the main study. A T4 optionally checks synthetic PyTorch mapping/counts under changed-input CUDA graph replay. It cannot run the BF16 NVFP4 Marlin benchmark.

Run All with defaults. No internet, installs, model downloads, SGLang checkout, or paid GPU needed.
The final cell exports a ZIP, including failures. Return the ZIP for analysis and stop the GPU session if enabled.

The H200/SM80–SM90 full local MoE harness is a separate `marlin_bench.py` file. No serving source is changed or submitted here.
""",
)
cell(
    "code",
    "from pathlib import Path\nimport json, time, traceback, zipfile\nfrom IPython.display import FileLink, display\n"
    "RUN_CUDA_ROUTE_CHECKS = True  # skipped automatically without CUDA\n"
    "SEEDS = 4\n"
    "run_dir = Path.cwd() / ('marlin_tiles_01_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))\n"
    "run_dir.mkdir(parents=True, exist_ok=False)\n"
    "support = run_dir / 'tile_study.py'\n"
    f"support.write_text({source!r}, encoding='utf-8')\n"
    "ns = {'__name__': 'tile_study_embedded'}\n"
    "exec(compile(support.read_text(encoding='utf-8'), str(support), 'exec'), ns)\n",
)
cell(
    "code",
    """try:
    summary = ns['structure_study'](run_dir, SEEDS)
    print(json.dumps(summary, indent=2))
    if RUN_CUDA_ROUTE_CHECKS:
        try:
            print(json.dumps(ns['cuda_route_checks'](run_dir), indent=2))
        except ImportError as exc:
            (run_dir / 'cuda_route_checks.json').write_text(json.dumps({'status': 'skipped', 'reason': str(exc)}))
except Exception:
    (run_dir / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    print(traceback.format_exc())
finally:
    ns['manifest'](run_dir)
    archive = run_dir.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(run_dir.iterdir()):
            if path.is_file():
                zipped.write(path, path.name)
    print('Download the whole ZIP. Counts are not speedups.')
    display(FileLink(str(archive)))
""",
)
notebook = dict(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    },
    nbformat=4,
    nbformat_minor=5,
)
for i, c in enumerate(cells):
    c["id"] = f"tile-study-{i}"
target = root / "01_marlin_tile_structure.ipynb"
target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(target)
