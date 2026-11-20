"""Independent source binding and abstract protocol counterexamples, CPU only."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent;serving=here.parents[1]
p=here.parent/'coordination/finalize_reduce/protocol.cu'
n=serving/'sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh'
def body(text,name):
 start=text.index(name);start=text.index('{',start);depth=1;i=start+1
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[start+1:i-1]
old=body(n.read_text(),'void all_reduce_1shot_push_kernel(')
new=body(p.read_text(),'void transport_body(')
new=new[new.index('  using namespace device;'):]
old=old[old.index('  using namespace device;'):]
expected=old.replace('vec.load(params.input, vid);','fill_vec<Fused, HasScale>(vec, params.a, vid);')
assert new.strip()==expected.strip(),'Protocol body differs beyond producer replacement'

def partial_counter_grid(full_flip):
 # One representative payload vector, formerly processed by block0's stride,
 # is processed by block1 in the next larger grid.
 old_values=list(range(1,9));expected=sum(old_values)
 inbox={0:old_values.copy(),1:[None]*8}
 counters=[1,1 if full_flip else 0] # Fast rank0 finishedoldround.
 phase=counters[1]&1
 inbox[phase][0]=100 # Fast rank0 pushes next call before slowrank1 drainsoldcall.
 slow_old=sum(inbox[0])
 # Slowrank1 clears its old phase afterreading. Reused newarrival disappears.
 inbox[0]=[None]*8
 lost=inbox[phase][0] is None
 return dict(all_counters_flipped=full_flip,fast_rank_next_vector_phase=phase,
             slow_rank_old_sum=slow_old,expected_old_sum=expected,
             next_rank0_arrival_lost=lost)
a=partial_counter_grid(False);b=partial_counter_grid(True)
assert a['slow_rank_old_sum']!=a['expected_old_sum'] and a['next_rank0_arrival_lost']
assert b['slow_rank_old_sum']==b['expected_old_sum'] and not b['next_rank0_arrival_lost']
# Two CUDAstreams can both read epoch0 before either sets1. A hostenqueue lock
# does not make async execution nonoverlapping when the streams differ.
stream_epoch={'A':0,'B':0};slot=[1,2,3,4,5,6,7,8]
slot[0]=100 # B overwrites A's rank0payload before A consumes it.
concurrent=dict(initial_epochs=stream_epoch,old_round_sum=sum(slot),expected_old_round_sum=36,requires_device_execution_order=True)
assert concurrent['old_round_sum']!=concurrent['expected_old_round_sum']
# kAtom4 means a pair of BF16 +zeros, not each scalarzero independently.
words=[0,0x00008000,0x80000000,0x3f800000,0x00003f80,0x7fc00000]
encoded=[0x8000 if w==0 else w for w in words]
assert all(w!=0 for w in encoded) and encoded[3:]==words[3:]
r=dict(scope='Source identity and abstract asynchronous message trace, not CUDA/NVLink execution',native_sha256=hashlib.sha256(n.read_bytes()).hexdigest(),prototype_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),body_exact_except_local_vector_producer=True,
 partial_counter_grid_counterexample=a,full_counter_grid_control=b,concurrent_stream_counterexample=concurrent,
 zero_atom_examples=dict(input_words=words,encoded_words=encoded,kAtom_bytes=4),
 no_prototype_defect_claim='Current virtual fixture keeps a fixed completecontroller grid and serialized execution. Counterexamples target unsafe future admission/integration.')
(here/'finalizer_protocol_review.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
