"""Patch the existing non-DP, single-host request-poll broadcast only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

here=Path(__file__).resolve().parent
root=here.parents[1]
manifest=json.loads((here.parent/"ssengine_graph_prep/manifest.json").read_text())
base=manifest["base"]
parent=manifest["parent_tree"]
path="python/sglang/srt/managers/scheduler_components/request_receiver.py"
group_path="python/sglang/srt/distributed/parallel_state.py"
mq_path="python/sglang/srt/distributed/device_communicators/shm_broadcast.py"
common_path="python/sglang/srt/utils/common.py"
t=Path(os.environ.get("TEMP",os.environ.get("TMPDIR","/tmp")))
with tempfile.TemporaryDirectory(dir=t) as td:
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(td)/"index"))
    def git(*args,data=None):
        return subprocess.check_output(["git","-C",str(root/"sglang"),*args],input=data,env=env)
    def get(p):
        return git("show",parent+":"+p).decode()
    def single_method(s,name):
        items=[i for i in ast.walk(ast.parse(s)) if isinstance(i,ast.FunctionDef) and i.name==name]
        assert len(items)==1
        return ast.get_source_segment(s,items[0])
    code=get(path)
    needle="""            if get_parallel().tp_size != 1:
                tp_group = get_parallel().tp_group
                recv_reqs = broadcast_pyobj("""
    assert code.count(needle)==1
    new=code.replace(needle,"""            if get_parallel().tp_size != 1:
                tp_group = get_parallel().tp_group
                queue = tp_group.mq_broadcaster
                # An empty poll on a single host can use the existing TP ring.
                # Preserve the ordinary payload protocol for all arrivals.
                if queue is not None and queue.handle.remote_subscribe_port is None:
                    if not queue.broadcast_object(bool(recv_reqs)):
                        return []
                recv_reqs = broadcast_pyobj(""")
    compile(new,path,"exec")
    git("read-tree",parent)
    blob=git("hash-object","-w","--stdin",data=new.encode()).decode().strip()
    git("update-index","--cacheinfo","100644",blob,path)
    tree=git("write-tree").decode().strip()
    inc=git("diff","--binary",parent,tree)
    cum=git("diff","--binary",base,tree)
    git("read-tree",base)
    git("apply","--cached","--whitespace=nowarn","-",data=cum)
    assert git("write-tree").decode().strip()==tree
    (here/"candidate.diff").write_bytes(cum)
    (here/"incremental.diff").write_bytes(inc)
    (here/"shm_reference.py").write_text(get(mq_path),encoding="utf8")
    for name,s in [("parent",code),("candidate",new)]:
        (here/(name+"_receive.py")).write_text(single_method(s,"_broadcast_reqs_across_ranks"),encoding="utf8")
    (here/"baseline_broadcast.py").write_text(single_method(get(common_path),"broadcast_pyobj"),encoding="utf8")
    (here/"same_node.py").write_text(single_method(get(group_path),"in_the_same_node_as"),encoding="utf8")
    m={"base":base,"parent_sha256":manifest["parent_sha256"],"tree":tree,
       "candidate_sha256":hashlib.sha256(cum).hexdigest(),"source":path,"transport_source":mq_path}
    (here/"manifest.json").write_text(json.dumps(m,indent=2))
    print(m)
