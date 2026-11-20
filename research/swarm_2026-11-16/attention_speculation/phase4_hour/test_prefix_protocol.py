"""Offline causal-protocol checks; this does not prove neural model numerics."""
import hashlib, json, random
from pathlib import Path
from prefix_protocol import *
P=Path(__file__).resolve().parent
rng=random.Random(8031)
checks=0; all_accepted=0; reused=0; cases=[]
for width in SUPPORTED:
    for wrong in range(8):
        for count in (1,2,7,8,9,33,65):
            prompt=[rng.randrange(31) for _ in range(rng.randrange(1,12))]
            full=full_draft8(prompt,wrong)
            assert pack_prefix([full],width)[0] == full[:width]
            result=run_request(prompt,count,[width],[wrong])
            checks+=1; all_accepted+=result['all_accepted']; reused+=result['reused_slots']
for n in range(400):
    prompt=[rng.randrange(31) for _ in range(rng.randrange(1,20))]
    count=rng.randrange(1,100)
    widths=[rng.choice(SUPPORTED) for _ in range(9)]
    mismatches=[rng.randrange(8) for _ in range(11)]
    eos=rng.randrange(31) if n%2 else None
    physical={i:('previous_request',i) for i in range(150)} if n%3==0 else None
    result=run_request(prompt,count,widths,mismatches,eos,physical)
    checks+=1; all_accepted+=result['all_accepted']; reused+=result['reused_slots']
# Exhaust exact first-mismatch / all-accepted semantics, including output tail.
for width in SUPPORTED:
    for accepted in range(width):
        candidates=list(range(100,100+width))
        top1=candidates[1:]+[999]
        if accepted<width-1: top1[accepted]=777
        c=accept_prefix(candidates,top1,257)
        assert c.accepted==accepted and c.committed==accepted+1
        assert c.bonus==top1[accepted] and c.new_seq_len==258+accepted
        assert c.emitted==tuple(candidates[1:accepted+1])+ (top1[accepted],)
        assert c.input_tokens==tuple(candidates[:accepted+1]); checks+=1
# Negative lifetime and unverified-row guards must actually detect failures.
buf=AuxBuffer(); old=buf.write([1,2]); buf.write([3])
negative=[]
for name,action in [('overwritten_aux',lambda:buf.consume(old,1)),
                    ('unverified_hidden',lambda:buf.consume(buf.generation,2)),
                    ('regenerated_short_draft',lambda:pack_prefix([[1,2,3,4]],4))]:
    try: action()
    except (ValueError,RuntimeError): negative.append(name)
    else: raise AssertionError(name)
assert reused>0 and all_accepted>0
report={'status':'PASS','request_and_accept_checks':checks,'negative_guards':negative,
        'all_accepted_cycles':all_accepted,'physical_slot_reuses':reused,
        'scope':'Integer causal protocol only; toy full8 draft proposals generated before prefix selection. No model or GPU arithmetic proof.',
        'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'prefix_protocol.py',Path(__file__)]}}
(P/'cpu_protocol_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
