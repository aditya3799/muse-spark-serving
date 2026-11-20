"""Resume yesterday's interrupted true-width draft-prefill fixture without overwrites."""
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'research_2026-11-18_window_repair'
BASE = '8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'
assert hashlib.sha256((ROOT/'release_2026-11-18/candidate.diff').read_bytes()).hexdigest() == BASE
os.environ['OKC_SMALL_WINDOW_REPLAY']='1'
os.environ['OKC_EARLY_TAPS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,str(OLD))
source=(OLD/'gpu_final.py').read_text()
# Imports and native source references stay anchored at the original fixture;
# only receipts move to the new research directory. Preserve all timing arms.
source=source.replace('SEEDS={"pilot":915137,"holdout":715631}',
                      'SEEDS={"pilot":392617,"holdout":874953}')
source=source.replace('(HERE/f"{LABEL}_{stage}', '(OUTPUT/f"window_resume_{stage}')
source=source.replace('B=1\n', 'B=1\nOUTPUT=Path('+repr(str(HERE))+')\n')
source=source.replace('import torch,os', 'import torch,os\ntorch.set_num_threads(1)')
source=source.replace('detail=dict(seed=seed,orders=[],cases={},complete=False)',
                      'detail=dict(seed=seed,orders=[],cases={},complete=False, source_sha256='+repr(hashlib.sha256((OLD/'gpu_final.py').read_bytes()).hexdigest())+')')
# Fail at a first observed value disagreement instead of spending a GPU slot on
# a candidate already falsified. Warmup is also checked before timed rounds.
source=source.replace('if step>=0:\n                base=snapshot',
    'if step==-1:\n                assert not any(warm[scene]["wrong"].values()), warm[scene]\n            if step>=0:\n                base=snapshot')
source=source.replace('bad==0\n', 'bad==0\n                assert receipt["correct"], d\n')
exec(compile(source,str(OLD/'gpu_final.py'),'exec'), {'__name__':'__main__','__file__':str(OLD/'gpu_final.py')})
