from pathlib import Path
H=Path(__file__).resolve().parent
p=H/"tree_dispatch.py";s=p.read_text().replace("from sglang.srt.runtime_context import get_parallel","from sglang.srt.runtime_context import get_parallel\nfrom sglang.srt.layers.dp_attention import get_dp_tp_group")
s=s.replace("tp_group=get_parallel().tp_group","tp_group=get_dp_tp_group()")
s=s.replace("    worker._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY,prediction)",'''    tp_group=get_dp_tp_group()
    if tp_group.world_size>1 and not worker._tp_sync.enabled(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY):
        tp_group.broadcast(prediction,src=0)
    else:
        worker._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY,prediction)''')
p.write_text(s)
p=H/"test_tree_dispatch.py";s=p.read_text().replace("scope=dict(get_parallel=lambda:par,","scope=dict(get_parallel=lambda:par,get_dp_tp_group=lambda:par.tp_group,")
s=s.replace('w._tp_sync=N(sync=','w._tp_sync=N(enabled=lambda site:True,sync=')
p.write_text(s)

