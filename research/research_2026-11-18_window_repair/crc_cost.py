"""Collision-conservative CRC fingerprint alternative; CPU comparison only."""
from array import array
import hashlib,json,statistics,struct,zlib
from pathlib import Path
from regions import DFlashRegionCoverage
import scheduler_cost,stress
class QuickCoverage(DFlashRegionCoverage):
    def _indexes(self,ids,begin,namespace):
        if not isinstance(ids,array) or ids.typecode!="q":ids=array("q",ids[:begin])
        h=zlib.crc32(json.dumps(namespace,separators=(",",":")).encode())
        h=zlib.crc32(memoryview(ids)[:begin].cast("B"),h)
        a,b=struct.unpack("<II",hashlib.blake2s(struct.pack("<QI",begin,h),digest_size=8).digest())
        n=8*len(self.bits);b|=1
        return ((a+j*b)%n for j in range(3))
if __name__=="__main__":
    scheduler_cost.DFlashRegionCoverage=QuickCoverage;stress.DFlashRegionCoverage=QuickCoverage
    result={str(b):scheduler_cost.bench(b,16384//b,rounds=9) for b in (1,16)}
    result["physical"]=stress.physical()
    H=Path(__file__).resolve().parent;(H/"crc_cost.json").write_text(json.dumps(result,indent=2))
    print({k:v["median"] for k,v in result.items() if k!="physical"},result["physical"])

