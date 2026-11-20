"""Check TP subgroup source global rank, post-receive callbacks, and fallback paths."""
import ast
from pathlib import Path
import subprocess
import textwrap
from trial import *

def stress(rank,world,path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+path,rank=rank,world_size=world)
    groups=[list(range(4)),list(range(4,8))]
    pgs=[dist.new_group(ranks=g,backend="gloo") for g in groups]
    ix=rank//4
    ranks=groups[ix]
    pg=pgs[ix]
    queue=MessageQueue.create_from_process_group(pg,1<<22,6)
    assert queue._is_writer==(rank==ranks[0])
    assert dist.get_process_group_ranks(pg)[0]==ranks[0]
    group=SimpleNamespace(cpu_group=pg,rank=rank,ranks=ranks,mq_broadcaster=queue)
    parallel=SimpleNamespace(pp_rank=0,attn_dp_enabled=False,tp_size=4,tp_group=group)
    original=source_func("baseline_broadcast.py","broadcast_pyobj",dict(torch=torch,dist=dist,np=np,pickle=pickle,is_musa=lambda:False))
    scope=dict(get_parallel=lambda:parallel,broadcast_pyobj=original)
    arms=[source_func(f+"_receive.py","_broadcast_reqs_across_ranks",dict(scope)) for f in ("parent","candidate")]
    manifest=json.loads((here/"manifest.json").read_text())
    code=subprocess.check_output(["git","-C",str(here.parents[1]/"sglang"),"show",manifest["base"]+":"+manifest["source"]]).decode()
    nodes=ast.walk(ast.parse(code))
    fn=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=="recv_requests")
    fn.decorator_list=[]
    ns=dict(get_parallel=lambda:parallel)
    exec("from __future__ import annotations\n"+ast.unparse(fn),ns)
    loop=ns["recv_requests"]
    calls=[]
    inputs=[([],[]),([],["source-abort"]),([{"sequence":int(ranks[0])}],[{"abort":rank}]),
            (["A","B","C"],[])]
    scenarios=[]
    # A simulated remote-transport handle must bypass the ring on every rank.
    # The original Gloo payload then executes against a real subgroup.
    fake_remote=SimpleNamespace(handle=SimpleNamespace(remote_subscribe_port=333),
                                broadcast_object=lambda *unused:unavailable())
    for name,ring in (("local",queue),("disabled",None),("remote_handle",fake_remote)):
        group.mq_broadcaster=ring
        for i,(arr,local) in enumerate(inputs):
            expected=arr+local if rank==ranks[0] else None
            official=[expected]
            dist.broadcast_object_list(official,src=ranks[0],group=pg)
            for arm_index,arm in enumerate(arms):
                calls.clear()
                recv=SimpleNamespace(scripted_scheduler_hook=None,recv_skipper=None,input_blocker=None,
                     _pull_raw_reqs=lambda:arr if rank==ranks[0] else None,
                     _broadcast_reqs_across_ranks=lambda data,loc:arm(None,data,loc),
                     unwrap_pickle_wrapper=lambda x:calls.append(("unwrap",tuple(x))),
                     # This simulates a pending MM request becoming available on an empty poll.
                     _apply_mm_receiver=lambda x:(calls.append(("mm",tuple(x))) or (x+["pending-MM"])),
                     _finalize_shm_features=lambda x:calls.append(("shm",tuple(x))))
                got=loop(recv,local_reqs=local if rank==ranks[0] else ["ignored-on-nonleader"])
                assert got==official[0]+["pending-MM"]
                assert calls==[("unwrap",tuple(official[0])),("mm",tuple(official[0])),("shm",tuple(got))]
            scenarios.append((name,i))
    dist.barrier()
    if rank==0:
        (here/"subgroups.json").write_text(json.dumps(dict(world=world,groups=groups,global_writers=[0,4],
          arms=["parent","candidate"],cases_per_rank_per_arm=len(scenarios),ranks_per_group=4,
          remote_handle_gloo_fallback="simulated all-rank remote handle with real Gloo payload",
          disabled_fallback="actual Gloo payload",exact_payloads=True,empty_post_receive_hooks_executed=True,
          local_abort_added_on_tp_group_writer=True,nonleader_list_ignored_as_in_parent=True),indent=2))
        print("passed source selection 0 and 4, 12 cases per rank/arm")
    dist.destroy_process_group()

if __name__=="__main__":
    with tempfile.TemporaryDirectory() as tmp:
        resource_tracker.ensure_running()
        mp.start_processes(stress,args=(8,str(Path(tmp)/"store")),nprocs=8,join=True,start_method="fork")
