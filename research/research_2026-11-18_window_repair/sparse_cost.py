"""Bounded conservative sparse key plus test deliberately colliding unequal prefixes."""
from array import array
import hashlib,json,struct
from pathlib import Path
from regions import DFlashRegionCoverage
import stress,scheduler_cost
class SparseCoverage(DFlashRegionCoverage):
    def _indexes(self,ids,begin,namespace):
        if not isinstance(ids,array) or ids.typecode!="q":ids=array("q",ids[:begin])
        h=hashlib.blake2b(digest_size=16,person=b"dflash-sparse-v1")
        h.update(json.dumps(namespace,separators=(",",":")).encode())
        h.update(struct.pack("<Q",begin))
        view=memoryview(ids)
        for k in range(5):
            start=max(0,begin-32)*k//4
            h.update(view[start:min(begin,start+32)].cast("B"))
        a,b=struct.unpack("<QQ",h.digest());b|=1;n=8*len(self.bits)
        return ((a+j*b)%n for j in range(3))
def collision():
    cov=SparseCoverage(2048,1)
    a=array("q",range(74160));b=array("q",a)
    b[777]+=9232
    one,two=stress.request(a,"x"),stress.request(b,"x")
    assert cov.prepare([one],[16384],[16384])
    b=b[:32521];two=stress.request(b,"x")
    limited=cov.match_limit(two,b,len(b)-1)
    assert limited==len(b)-2048
    return {"first_divergence":777,"unmodified_length":32521,"conservative_limit":limited,"natural_hit":777}
if __name__=="__main__":
    scheduler_cost.DFlashRegionCoverage=SparseCoverage;stress.DFlashRegionCoverage=SparseCoverage
    result={str(b):scheduler_cost.bench(b,16384//b,rounds=9) for b in (1,16)}
    result["forced_unsampled_divergence"]=collision();result["physical"]=stress.physical()
    for k,v in stress.adversarial()["1"].items():result.setdefault("ordering",{})[k]=v["regions"]["totals"]
    (Path(__file__).parent/"sparse_cost.json").write_text(json.dumps(result,indent=2))
    print({k:v["median"] for k,v in result.items() if k in ("1","16")},result["physical"],result["forced_unsampled_divergence"])

