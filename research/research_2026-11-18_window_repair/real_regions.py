from array import array
from collections import Counter
import hashlib,json,statistics,sys
from pathlib import Path
from types import SimpleNamespace as S
from regions import DFlashRegionCoverage
H=Path(__file__).resolve().parent
PUB=H.parent/"research_2026-11-18_frontier"
sys.path.insert(0,str(PUB))
from muse_records import samples
from tokenizers import Tokenizer
def run():
    checkpoint=PUB/"muse_tokenizer.json"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=="ac2f41ef2444dec82fdc90e334f80543f298535005fe78aec7e8311ef86f84d0"
    tok=Tokenizer.from_file(str(checkpoint))
    all_counts={}
    for chunk in (8192,16384):
        for retention in ("cold","prompt","continuation"):
            agg=Counter();turns=[]
            for trace in samples(joined=True):
                hist=array("q");coverage=DFlashRegionCoverage(2048,1);tip=0;native_tip=0
                for msg in trace:
                    # One out-of-vocabulary integer separates public message boundaries.
                    hist.append({"user":-1,"assistant":-2,"tool":-3}[msg["role"]])
                    ids=tok.encode(msg["content"],add_special_tokens=False).ids
                    if msg["role"]!="assistant":
                        hist.extend(ids);continue
                    r=S(origin_input_ids=hist,full_untruncated_fill_ids=hist,extra_key=None,cache_salt=None,
                        session=None,multimodal_inputs=None,positional_embed_overrides=None,skip_radix_cache_insert=False,output_ids=[])
                    end=len(hist);ordinary=min(tip,end-1) if retention!="cold" else 0
                    native_hit=min(native_tip,end-1) if retention!="cold" else 0
                    hit=coverage.match_limit(r,hist,ordinary);omitted=0
                    for start in range(hit,end,chunk):
                        n=min(chunk,end-start)
                        if coverage.prepare([r],[start],[n]):omitted+=n
                    cached=end+len(ids) if retention=="continuation" else end
                    hist.extend(ids)
                    native_tip=max(native_tip,cached)
                    if coverage.may_publish(r,cached):tip=max(tip,cached)
                    agg.update(native_target=end-native_hit,target=end-hit,materialized=end-hit-omitted,omitted=omitted,turns=1,
                        benefited=int(omitted>0),extra_target=native_hit-hit)
                    turns.append(omitted)
            all_counts[f"{chunk}/{retention}"]=dict(agg)
            print(chunk,retention,dict(agg),flush=True)
    (H/"real_region_replay.json").write_text(json.dumps(all_counts,indent=2)+"\n")
if __name__=="__main__":run()


