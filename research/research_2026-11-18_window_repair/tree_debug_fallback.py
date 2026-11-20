from pathlib import Path
H=Path(__file__).resolve().parent
p=H/"tree_dispatch.py";s=p.read_text().replace("from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite","from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite\nfrom sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN").replace("return (worker.lilicorr is None","return (SIMULATE_ACC_LEN <= 0 and worker.lilicorr is None");p.write_text(s)
for fn,name in [("test_tree_dispatch.py","scope"),("tree_tp_gloo.py","env")]:
    p=H/fn;s=p.read_text().replace(name+"=dict(",name+"=dict(SIMULATE_ACC_LEN=0,",1)
    if fn=="test_tree_dispatch.py":
        s=s.replace("assert choose(w,b,prefix,slots,anchors,False) is None;checks+=1","assert choose(w,b,prefix,slots,anchors,False) is None;checks+=1\nscope['SIMULATE_ACC_LEN']=4\nassert active() is None;checks+=1\nscope['SIMULATE_ACC_LEN']=0")
    p.write_text(s)

