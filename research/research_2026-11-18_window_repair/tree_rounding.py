"""Explicit E4M3 values and zero query give uniform attention; show BF16 split-rounding."""
import itertools,json,struct
from pathlib import Path
def bf(x):
    bits=struct.unpack("<I",struct.pack("<f",x))[0]
    up=(bits+0x7fff+((bits>>16)&1))>>16
    return struct.unpack("<f",struct.pack("<I",up<<16))[0]
def main():
    for p in itertools.combinations_with_replacement([-.5,0.,.125,.25,.5,.75,1.,2.,4.,8.],3):
        for s in ([0.],[.125],[.25],[1.]):
            a=sum(p)/len(p);b=sum(s)/len(s)
            one=bf((sum(p)+sum(s))/(len(p)+len(s)))
            merged=bf((len(p)*bf(a)+len(s)*bf(b))/(len(p)+len(s)))
            if one!=merged:
                out=dict(prefix_values=p,suffix_values=s,q=0,exact_prefix_mean=a,returned_prefix_bf16=bf(a),
                    one_softmax_bf16=one,weighted_merge_of_returned_partial_outputs_bf16=merged,
                    scope="Uniform exact weights, BF16-rounded partial outputs; explicit E4M3-exact KV values. Illustrates two-stage rounding possibility. Native FA3 execution is unmeasured.")
                (Path(__file__).parent/"fixed_tree/rounding.json").write_text(json.dumps(out,indent=2))
                print(out);return
if __name__=="__main__":main()

