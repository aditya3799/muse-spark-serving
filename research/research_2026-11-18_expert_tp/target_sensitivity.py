"""Four-load sensitivity and exact payload audit for fused pooler/router transport."""
import json
import re
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SERVE=HERE.parent
ROOT=SERVE.parent
LOADS=(1,4,16,32)
def read(relative):
    return json.loads((ROOT/relative).read_text(encoding="utf-8"))
def time_us(row):
    return row["baseline_ns"]/1000,row["median_ns"]/1000
def breakdown(report):
    result={key:0.0 for key in ("expert","attention","dense","glue")}
    for line in report.splitlines():
        match=re.search(r"\s+([\d.]+)\s+\S*s\s+[\d.]+%\s+\d+\s*$",line)
        if not match: continue
        name=line[:match.start()].lower()
        group="expert" if "moe_fused" in name else "attention" if "attn_kernel" in name else "dense" if any(t in name for t in ("sk2_kernel","void rq<","nvjet","cgemv")) else "glue"
        result[group]+=float(match.group(1))
    return result

def payload_fixture():
    # The multicast protocol sanitizes each 16-bit zero, so FP32 logits require a reversible nonzero encoding.
    rng=np.random.default_rng(2192)
    special=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc03eba,0x3f800000,0x00008000,1,0xffffffff],dtype="uint32")
    raw=rng.integers(0,2**32,size=(8,8,32),dtype="uint32")
    raw.ravel()[:len(special)]=special
    naive=raw.view("uint16").copy()
    naive[naive==0]=0x8000
    false_logits=int(np.sum(naive.view("uint32")!=raw))
    assert false_logits
    # Low/next 15 bits and high 2 bits, all shifted away from zero. Store raw u16: no FP conversion.
    code=np.stack([1+(raw&32767),1+((raw>>15)&32767),1+(raw>>30)],axis=-1).astype("uint16").reshape(8,8,96)
    assert code.min()>0
    # Full pooler 4096 BF16 features, local 512; local 32 logits x 3 limbs = 96, whole rank chunk 608 u16.
    pool=rng.integers(0,0x7f00,size=(8,8,512),dtype="uint16")
    pool[:,0,0]=0
    local=np.concatenate([pool,code],axis=-1)
    assembled=local.transpose(1,0,2).copy()
    assembled[assembled==0]=0x8000
    out=assembled[:,:,512:].reshape(8,8,32,3).astype("uint32")-1
    out=out[:,:,:,0] | (out[:,:,:,1]<<15) | (out[:,:,:,2]<<30)
    assert np.array_equal(out,raw.transpose(1,0,2))
    got_pool=assembled[:,:,:512].copy()
    want_pool=pool.transpose(1,0,2).copy()
    assert np.array_equal(got_pool,np.where(want_pool==0,0x8000,want_pool))
    # Negative control: swapping two rank columns must be caught by the independent layout.
    mixed=out.copy(); mixed[:,[0,1]]=mixed[:,[1,0]]
    assert not np.array_equal(mixed,raw.transpose(1,0,2))
    return dict(checked_logits=out.size,checked_pooler_values=pool.size,naive_corrupt_logits=false_logits,
                encoded_bit_errors=0,local_width=608,traffic_uplift=608/512-1)

def main():
    parent=read("serving/release_2026-11-18/scored_submission_status.json")["result"]
    layer=read("serving/profiles/725c19cbeb9b4d48.json")
    attn=read("muse_attention_decode/profiles/submission_keell0renz_exact.e31d3f98ee8c446b.json")
    dense=read("muse_bf16_proj/profiles/submission_validated_routes_v139.caae25fa6f9d4db1.json")
    p0=read("serving/profiles/9b5340c0f17b4fd4.json")
    p1=read("muse_prefill_layer/profiles/a1d2f61ed3534773.json")
    saved={}
    for n in (1,16,32):
        phase=[]
        part={key:0 for key in ("expert","attention","dense","glue")}
        stock=donor=0.
        for bench,prof in zip(layer["benchmarks"],layer["profiles"]):
            c=bench["case"]
            if c["load"]!=f"verify {n}": continue
            weight=c["weight"]
            s,d=time_us(bench); stock+=weight*s/1000; donor+=weight*d/1000
            split=breakdown(prof["report"])
            assert sum(split.values())>80
            for k,v in split.items(): part[k]+=v*weight/1000
            phase.append(dict(layer=c["layer"],stock_us=s,donor_us=d,trace=split))
        assert len(phase)==2
        a_ms=sum((time_us(b)[0]-time_us(b)[1])*(46 if b["case"]["layer"]=="sliding" else 16)/1000 for b in attn["benchmarks"] if b["case"]["load"]==f"verify {n}")
        ds_ms={b["case"]["layer"]:(time_us(b)[0]-time_us(b)[1])/1000 for b in dense["benchmarks"] if b["case"]["load"]==f"verify {n}"}
        saved[n]=dict(stock_ms=stock,donor_ms=donor,saving_ms=stock-donor,donor_profile_gpu_ms=part,detail=phase,
                      separate_attention_component_saved_ms_per_62_layers=a_ms,
                      separate_dense_component_saved_ms_per_62_layers={k:v*(60 if k in ("router","unpooler","pooler") else 62) for k,v in ds_ms.items()})
    parent_r={r["concurrency"]:r for r in parent["rungs"]}
    frontier={p["concurrency"]:p["ratio"] for p in parent["frontier"]["points"]}
    compare=[]
    for a,b in zip(p0["benchmarks"],p1["benchmarks"]):
        assert a["case"]["load"]==b["case"]["load"] and a["case"]["layer"]==b["case"]["layer"]
        compare.append(dict(load=a["case"]["load"],layer=a["case"]["layer"],ctx=a["case"]["ctx"],
                            humming_ms=a["median_ns"]/1e6,fused_ms=b["median_ns"]/1e6,
                            ratio=a["median_ns"]/b["median_ns"]))
    pre=[r["ratio"] for r in compare if r["load"]=="first turn"]
    turn=[r["ratio"] for r in compare if r["load"]!="first turn"]
    scenarios=[]
    def estimate(transfer,net_us,first_gain,later_gain,c4):
        gains=[]
        data=[]
        for n in LOADS:
            r=parent_r[n]
            cycle=1000*r["acceptance"]/r["tok_s_per_user_p50"]
            # 4-user q=8 is absent on the decoder board. Vary its own net saving instead of copying another rung ratio.
            delta=(saved[1]["saving_ms"]+c4*(saved[16]["saving_ms"]-saved[1]["saving_ms"]) if n==4 else saved[n]["saving_ms"])
            # Additional critical-path cost vs parent, per layer, attending to two communication seams.
            dd=transfer*delta-62*2*net_us/1000
            decode_gain=cycle/(cycle-dd)
            # Phase-duration proxy, explicitly mixing marginal medians and p90s. No official frontier interpolation here.
            first=r["ttft_first_p90_s"]; later=12*r["ttft_later_p90_s"]; dec=13*220/r["tok_s_per_user_p50"]
            gain=(first+later+dec)/(first/first_gain+later/later_gain+dec/decode_gain)
            gains.append(gain)
            data.append(dict(users=n,decode_gain_proxy=decode_gain,first_fraction=first/(first+later+dec),
                             later_fraction=later/(first+later+dec),generation_fraction=dec/(first+later+dec),
                             modeled_gain_over_parent=gain,modeled_ratio_to_stock=gain*frontier[n]))
        return dict(transfer=transfer,net_added_us_per_seam=net_us,first_gain=first_gain,later_gain=later_gain,c4_interpolation=c4,
                    ratio_to_stock=parent["frontier"]["score"]*np.prod(gains)**.25,loads=data)
    for t in (.25,.5,.75,1):
        for c in (0.,2.,8.):
            scenarios.append(estimate(t,c,float(np.median(pre)),float(np.median(turn)),.2))
    scenarios.append(estimate(1.,0.,max(pre),max(turn),1.))
    for n in LOADS:
        assert parent_r[n]["acceptance"]>0
    result=dict(parent_scenario_assumptions="13 turns x 220 output tokens, p90 first/later TTFT and p50 generation. C4 saving interpolates between measured C1/C16. Full-layer stock delta times transfer coefficient stands in for unmeasured incumbent delta. Absolute times on separate workloads, identity collectives. Incremental seam overhead only, measured parent already includes collectives. These are sensitivities, not serving forecasts.",
                kernel_receipts=dict(decoder="725c19cbeb9b4d48",attention="e31d3f98ee8c446b",dense="caae25fa6f9d4db1",prefill_old="9b5340c0f17b4fd4",prefill_new="a1d2f61ed3534773"),
                layer=saved,prefill=compare,models=scenarios,payload=payload_fixture())
    (HERE/"target_sensitivity.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__": main()
