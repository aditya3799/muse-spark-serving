"""Replay the bounded conservative sparse fingerprint on real visible Muse step text."""
from pathlib import Path
from array import array
from types import SimpleNamespace as S
from sparse_cost import SparseCoverage
from regions import DFlashRegionCoverage
import stress,json
H=Path(__file__).resolve().parent
if __name__=="__main__":
    for cls in (DFlashRegionCoverage,SparseCoverage):
        a=array("q",range(74160));b=array("q",a);b[777]+=11
        cov=cls(2048,1)
        assert cov.prepare([stress.request(a,"t")],[16384],[16384])
        # An independently populated b canonical prefix already exists. A
        # collision may force its intrinsically safe full hit to be shortened.
        end=32521;b=b[:end]
        cutoff=cov.match_limit(stress.request(b,"t"),b,end-1)
        if cls is SparseCoverage:assert cutoff==end-2048
        else:assert cutoff==end-1
    source=(H/"real_regions.py").read_text(encoding="utf-8-sig")
    source=source.replace("from regions import DFlashRegionCoverage","from sparse_cost import SparseCoverage as DFlashRegionCoverage").replace("real_region_replay.json","sparse_region_replay.json")
    exec(compile(source,"real_regions.py","exec"),{"__name__":"__main__","__file__":str(H/"real_regions.py")})

