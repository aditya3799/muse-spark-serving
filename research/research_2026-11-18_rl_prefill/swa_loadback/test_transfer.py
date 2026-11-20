"""Use the actual pinned split, component redistributions and host-transfer builder on CPU."""
import ast
from array import array
from collections import defaultdict,namedtuple
import enum
import json
import random
from pathlib import Path
from types import SimpleNamespace as S,MethodType
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/"sglang/python/sglang/srt/mem_cache"
def get(path,owner,name,scope):
    t=ast.parse(path.read_bytes())
    if owner:t=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==owner)
    f=next(n for n in t.body if getattr(n,"name",None)==name)
    ns=scope.copy();exec(compile(ast.Module(ast.parse("from __future__ import annotations").body+[f],[]),str(path),"exec"),ns)
    return ns[name]
Key=get(ROOT/"radix_cache.py",None,"RadixKey",dict(array=array))
class Phase(enum.Enum):
    BACKUP_HOST=1;LOAD_BACK=2;BACKUP_STORAGE=3;PREFETCH=4
Result=namedtuple("Result","device_prefix_len host_hit_length swa_host_hit_length cache_actions")
def state():return S(lock_ref=0,host_lock_ref=0,session_ref=0,session_ids=None,metadata={},value=None,host_value=None)
scope=dict(torch=torch,CacheTransferPhase=Phase,PoolTransfer=S)
SWA=ROOT/"unified_cache/components/swa.py"; FULL=ROOT/"unified_cache/components/full.py"
redistribute_swa=get(SWA,"SWAComponent","redistribute_on_node_split",scope)
redistribute_full=get(FULL,"FullComponent","redistribute_on_node_split",scope)
transfer=get(SWA,"SWAComponent","build_hicache_transfers",scope)
split=get(ROOT/"unified_cache/unified_tree_core.py","UnifiedTreeCore","_split_node",dict(torch=torch,
    split_node_hash_value=lambda h,l,p:(None,None),UpdatePendingBackupOnNodeSplit=S,get_and_increase_time_counter=lambda:0))
check=get(HERE/"unified_radix_cache.py","UnifiedRadixCache","_match_tree",dict(RadixKey=Key,MatchPrefixParams=S))

class Core:
    def __init__(self,page,window,segments):
        self.page_size=page;self.window=window;self.next=0
        self.host_lru_lists={"swa":S(in_list=lambda n:False,insert_mru=lambda n:None,insert_after=lambda c,n:None)}
        self.has_swa_host_pool=True;self.enable_hicache=True;self.tlru_bookkeeping=False;self.is_host_memory_buffer_only=False
        self._for_each_component_lru=lambda *a,**k:None
        self._update_evictable_leaf_sets=self._update_duplicate_tracking=lambda *a:None
        self.swa=S(component_type="swa",tree_core=self,sliding_window_size=window,pool_name="swa")
        self.swa.redistribute_on_node_split=MethodType(redistribute_swa,self.swa)
        full=S(component_type="full");full.redistribute_on_node_split=MethodType(redistribute_full,full)
        self.components=[full,self.swa];self.root_node=self._new_node(0); self.root_node.parent=None
        self.last=self.root_node; ids=array("q",range(sum(n for n,tier in segments)))
        self.key=Key(ids,extra_key="alpha",cache_salt="tenant"); pos=0
        for n,tier in segments:
            node=self._new_node(0);node.key=self.key[pos:pos+n];node.parent=self.last
            self.last.children[node.key.child_key(page)]=node
            cd=node.component_data["swa"]
            vals=torch.arange(pos,pos+n)
            node.component_data["full"].value=vals.clone()
            cd.host_value=vals.clone() if tier=="host" else None
            cd.value=vals.clone() if tier=="device" else None
            self.last=node;pos+=n
        self.actions=[]
    def _new_node(self,priority):
        self.next+=1
        return S(id=self.next,priority=priority,children={},hit_count=0,external_cache_stored=False,creation_time=0,rotation_base=None,
                 load_back_pending_id=None,hash_value=None,event_hash_value=None,write_through_pending_id=None,
                 component_data={"swa":state(),"full":state()})
    def host_indices(self,node):
        out=transfer(self.swa,node,Phase.LOAD_BACK)
        return out[0].host_indices if out else torch.empty(0,dtype=torch.long)
    def match_prefix(self,params):
        key=params.key;offset=0;node=self.root_node; actions=[]
        while offset<len(key):
            child=node.children[key.child_key_at(offset,self.page_size)]
            n=min(len(key)-offset,len(child.key))
            if n<len(child.key):
                node,a=split(self,child.key,child,n)
                if a: actions.append(a)
                offset+=n;break
            offset+=n;node=child
        return Result(offset,0,len(self.host_indices(node)),actions)
    def full_order(self):
        values=[];sw=[];node=self.root_node
        while node.children:
            assert len(node.children)==1
            node=next(iter(node.children.values()));values.append(node.component_data["full"].value)
            cd=node.component_data["swa"];sw.append(cd.value if cd.value is not None else cd.host_value)
        return torch.cat(values),torch.cat(sw)

def test(page,w,segments,pending=False):
    core=Core(page,w,segments)
    if pending:core.last.write_through_pending_id=77
    def apply(acts):
        core.actions.extend(acts);acts.clear()
    wrapper=S(disable=False,_sliding_window_size=w,page_size=page,tree_core=core,cache_controller=True,is_mamba_enabled=False,
              linker=None,_apply_cache_actions=apply,_finalize_match_result=lambda params,result:result)
    params=S(key=core.key)
    prev=core.host_indices(core.last); result=check(wrapper,params);after=core.host_indices(core.last)
    length=len(core.key)
    assert result.device_prefix_len==length and result.swa_host_hit_length==len(after)
    assert torch.equal(prev[prev>=max(0,length-w)],after[after>=max(0,length-w)])
    assert len(after)<=len(prev)
    for ordered in core.full_order():assert torch.equal(ordered,torch.arange(length))
    if pending: assert len(core.actions)==1 and core.actions[0].ack_id==77
    twice=check(wrapper,params)
    assert result==twice and torch.equal(after,core.host_indices(core.last))
    return {"baseline_host_tokens":len(prev),"candidate_host_tokens":len(after),"visible_tokens":int((after>=length-w).sum()),
            "relocated_pending_writes":len(core.actions)}

def main():
    torch.set_num_threads(1)
    cases={"whole_host_chunk":test(1,2048,[(16384,"host")],True),
           "published_window_tail":test(1,2048,[(14336,"host"),(2048,"host")]),
           "recent_device_suffix":test(1,2048,[(16384,"host"),(220,"device")]),
           "device_resident":test(1,2048,[(16384,"device")]),
           "page64":test(64,2047,[(16384,"host"),(256,"device")])}
    for seed in range(100):
        rng=random.Random(seed);page=rng.choice([1,16,64]);w=rng.randint(16,256)
        segments=[(rng.randint(1,400)*page,rng.choice(["host","device"])) for _ in range(rng.randrange(1,8))]
        test(page,w,segments)
    (HERE/"transfer_results.json").write_text(json.dumps({"cases":cases,"random_cases":100,"source_scope":
        "Pinned Python _match_tree, _split_node, both component redistributions, and SWA build_hicache_transfers. Single-branch walk, LRU observers and TreeCore bindings supplied by the fixture. Each host index is a CPU torch tensor. No Rust binary or DMA measured."},indent=2))
    print(cases)
if __name__=="__main__":main()

