"""Independent pinned pack-layout transpose; no GEMM or runtime claim."""
from pathlib import Path
import ast, json, random, hashlib
P=Path(__file__).resolve().parent
oracle=P.parent/'aux_speculation/test_packed_addresses.py'
ns={};tree=ast.parse(oracle.read_text(encoding='utf-8-sig'))
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)],type_ignores=[]),str(oracle),'exec'),ns)
perm=[0,2,1,3,4,6,5,7]
def remap(word):
 return (word&0xf00ff00f)|((word&0x00f000f0)<<4)|((word&0x0f000f00)>>4)
def literal(word):
 return sum(((word>>(4*s))&15)<<(4*d) for d,s in enumerate(perm))
rng=random.Random(1927);words=[0,0xffffffff]+[1<<b for b in range(32)]+[rng.getrandbits(32) for _ in range(4096)]
for word in words:
 assert remap(word)==literal(word)
 assert remap(remap(word))==word
checks=0;shapes=[]
for n,k in [(64,64),(128,64),(64,128),(128,128),(256,64)]:
 plain=ns['reference'](n,k,4,False);transposed=ns['reference'](n,k,4,True)
 for physical,logical in plain.items():
  source=(physical//8)*8+perm[physical%8]
  assert transposed[source]==logical;checks+=1
 shapes.append([n,k])
r=dict(status='CPU_WORD_TRANSPOSE_PASS',permutation=perm,
 word_cases=len(words),pack_layout_checks=checks,shapes=shapes,
 bit_basis_argument='Masked shifts and OR use disjoint output bits; 32 single-bit basis cases prove equality to permutation for every uint32 word.',
 finite_precision='Codes and sign bits preserved exactly; no arithmetic performed.',
 scope='Pinned B4/A16 nonpacked-K interleave3 transpose flag only. Does not prove MMA swap-AB fragment loading, scale-to-lane mapping, accumulation order or GEMM numerical equivalence.',
 memory='In-register one-word transform would need no second persistent weight copy; instruction overhead and full kernel support unmeasured.',
 source_hashes={str(oracle):hashlib.sha256(oracle.read_bytes()).hexdigest()},
 warning='Pinned MMA8 branch alone appears to emit m8n8k8 BF16; it is not evidence of supported swap-AB. Chord source review required.')
(P/'WORD_TRANSPOSE_INDEPENDENT_PROOF.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(words=len(words),pack_layout_checks=checks)))
