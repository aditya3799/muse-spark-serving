"""Bound on additional cold coverage when the first global prefix region accepts poisoning."""
from pathlib import Path
from array import array
import json,sys
import stress
H=Path(__file__).resolve().parent
def variant():
    source=(H/"regions.py").read_text(encoding="utf-8-sig")
    for before,after in [(" or start<self.region",""),("not self.dirty or length<=self.region","not self.dirty or length<=0"),("first=max(self.region,length-self.window)","first=max(0,length-self.window)")]:
        assert source.count(before)==1
        source=source.replace(before,after)
    namespace={};exec(source,namespace);return namespace["DFlashRegionCoverage"]
def main():
    stable=stress.DFlashRegionCoverage;aggressive=variant()
    out={}
    original=array("q",range(74160));short=original[:753]
    extra={"long_then_short_repeat_32":[(original,"a")]+[(short,"a")]*32,"many_unrelated_shorts":[(original,"a")]+[(array("q",[10000+i])*753,"a") for i in range(32)],
           "many_shared_1k_prefix":[(original,"a")]+[(original[:1000]+array("q",[10000+i])*1024,"a") for i in range(32)],
           "same_prefix_other_salt":[(original,"a")]+[(short,"b")]*32}
    for name,cls in (("stable",stable),("first_region",aggressive)):
        stress.DFlashRegionCoverage=cls
        cases=stress.adversarial()["1"]
        base={n:{k:v[k]["totals"] for k in ("regions","native")} for n,v in cases.items()}
        for n,tr in extra.items():base[n]={k:stress.experiments(tr,1,k)["totals"] for k in ("regions","native")}
        out[name]=dict(cases=base,physical=stress.physical())
    for scene in out["stable"]["cases"]:
        row={}
        for name,v in out.items():
            d=v["cases"][scene]
            row[name]=[d["regions"]["target"]-d["native"]["target"],d["regions"]["omitted"]]
        print(scene,row)
    (H/"first_region_sensitivity.json").write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()

