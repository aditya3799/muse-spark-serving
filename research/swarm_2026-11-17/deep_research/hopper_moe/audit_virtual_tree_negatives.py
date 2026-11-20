from pathlib import Path
import json
P=Path(__file__).resolve().parent
p=lambda k:('p',k,0)
def canonical(t):
 if t[0]=='p':return t
 return ('+',)+tuple(sorted((canonical(t[1]),canonical(t[2]))))
correct=('+',('+',p(0),p(2)),('+',p(1),p(3)))
commuted=('+',('+',p(3),p(1)),('+',p(2),p(0)))
wrongbalanced=('+',('+',p(0),p(1)),('+',p(2),p(3)))
wrongsequential=('+',('+',('+',p(0),p(1)),p(2)),p(3))
assert canonical(correct)==canonical(commuted)
assert canonical(correct)!=canonical(wrongbalanced) and canonical(correct)!=canonical(wrongsequential)
r={'status':'reduction_grouping_negative_controls_passed','commutation_accepted':True,'balanced_wrong_pairing_rejected':True,'sequential_wrong_tree_rejected':True,'initial_strict_order_audit_failure':{'job':'w13_m8_s4_virtual','address_hex':'a720','instruction':'FADD.FTZ R25, R33, R25','observed':'Compiler commuted p0+p1 to p1+p0. Grouping and finite FP32 result are unchanged; NaN payload ordering is not established.'},'corrected_audit':'VIRTUAL_K_REDUCTION_SASS_AUDIT.json','scope':'Finite-value reduction grouping audit, not an all-bit-pattern or hardware numerical guarantee.'}
(P/'VIRTUAL_K_REDUCTION_NEGATIVE_CONTROLS.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
