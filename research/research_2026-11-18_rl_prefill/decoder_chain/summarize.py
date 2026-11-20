import argparse
import json
import math
import random
import statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("files",nargs="+",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    trials=[json.loads(f.read_text()) for f in args.files]
    result={"scope":"Single-rank four-block CUDA graph; paired adjacent AB/BA blocks are recomputed for each shape. Intervals represent local repeatability, not H200 end-to-end uncertainty.","runs":[]}
    for file,item in zip(args.files,trials):
        run={"source":file.name,"complete":item["complete"],"correct":item["correct"],"seed":item["seed"],"cases":{}}
        for shape,entry in item["bench"]["shapes"].items():
            times=entry["times"]
            if len(times["baseline"])<2:continue
            if len(times["baseline"])%2:continue
            rng=random.Random(item["seed"])
            # A pair includes both forward and reverse ordering.
            blocks=[[statistics.median(times[key][i:i+2]) for key in ("baseline","candidate","control")]
                    for i in range(0,len(times["baseline"]),2)]
            def ratio(rows,col):
                return math.exp(statistics.mean(math.log(r[0]/r[col]) for r in rows))
            central=ratio(blocks,1);control=ratio(blocks,2)
            bs=sorted(ratio(rng.choices(blocks,k=len(blocks)),1) for _ in range(5000))
            run["cases"][shape]={"estimate":central,"ci_95":[bs[124],bs[4874]],"control":control,
                                 "paired_blocks":len(blocks),"median_us":{k:statistics.median(v) for k,v in times.items()}}
        result["runs"].append(run)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=="__main__":main()
