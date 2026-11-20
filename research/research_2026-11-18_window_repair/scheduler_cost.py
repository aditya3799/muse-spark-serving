"""Synchronous scheduler-side cost of the actual bounded region policy."""
from array import array
import itertools,json,statistics,time
from pathlib import Path
from regions import DFlashRegionCoverage
from stress import request
H=Path(__file__).resolve().parent;W=2048
def bench(b,n,rounds=15):
    times={"cold":[],"tip":[],"middle_fork":[]};skips=[]
    for repeat in range(rounds):
        reqs=[request(array("q",(range(10097*i,10097*i+74160))),"tenant") for i in range(b)]
        cov=DFlashRegionCoverage(W,1);omitted=0
        begin=time.perf_counter_ns()
        for p in range(0,74160,n):
            es=[min(n,74160-p)]*b
            if cov.prepare(reqs,[p]*b,es):omitted+=sum(es)
            for r in reqs:cov.may_publish(r,p+es[0])
        times["cold"].append((time.perf_counter_ns()-begin)/1000)
        for name,end,prefix in (("tip",74913,74160),("middle_fork",33521,32768)):
            rs=[request(r.origin_input_ids[:prefix]+array("q",[123783])*753,"tenant") for r in reqs]
            begin=time.perf_counter_ns()
            hits=[cov.match_limit(r,r.origin_input_ids,prefix) for r in rs]
            for r,hit in zip(rs,hits):
                assert not cov.prepare([r],[hit],[end-hit])
                cov.may_publish(r,end)
            times[name].append((time.perf_counter_ns()-begin)/1000)
        skips.append(omitted)
    return dict(per_request_chunk=n,batch=b,omitted=skips,us=times,median={k:statistics.median(t) for k,t in times.items()})
if __name__=="__main__":
    result={str(b):bench(b,16384//b) for b in (1,16)}
    (H/"scheduler_cost.json").write_text(json.dumps(result,indent=2))
    print({k:(v["median"],v["omitted"][0]) for k,v in result.items()})

