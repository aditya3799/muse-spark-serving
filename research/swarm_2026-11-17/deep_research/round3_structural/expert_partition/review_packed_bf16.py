import ast,hashlib,json,random
from pathlib import Path
P=Path(__file__).resolve().parent;A=P.parent/'aux_speculation';ROOT=P.parents[3];H=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming'
files=[A/x for x in ['packed_staged_ops.py','probe_packed_native.py','test_packed_addresses.py','test_exponent_boundary.py']]+[H/x for x in ['kernel/humming.py','include/humming/kernel/process.cuh','include/humming/arith/exp_offset.cuh','include/humming/arith/epilogue_arith.cuh','include/humming/epilogue/smem_writer.cuh']]
source=(H/'kernel/humming.py').read_text();tree=ast.parse(source)
klass=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='HummingKernel');fn=next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=='check_dtype')
old=ast.get_source_segment(source,fn);new=old.replace('assert self.b_dtype.num_bits <= 8','assert self.b_dtype.num_bits <= 8 or (self.a_dtype == self.b_dtype == self.c_dtype == self.bs_dtype == dtypes.bfloat16 and self.is_channel_weight_scale and not self.has_zero_point and not self.has_bias and not self.use_f16_accum)')
assert old.count('assert self.b_dtype.num_bits <= 8')==1
x=ast.parse(old);y=ast.parse(new);xf=x.body[0];yf=y.body[0]
diffs=[i for i,(aa,bb) in enumerate(zip(xf.body,yf.body)) if ast.dump(aa)!=ast.dump(bb)]
assert len(diffs)==1 and len(xf.body)==len(yf.body)
i=diffs[0];assert isinstance(xf.body[i],ast.Assert) and isinstance(yf.body[i].test,ast.BoolOp)
assert ast.dump(yf.body[i].test.values[0])==ast.dump(xf.body[i].test)
# Import only pure CPU address functions, avoiding report side effects.
t=ast.parse((A/'test_packed_addresses.py').read_text());nodes=[v for v in t.body if isinstance(v,ast.FunctionDef)];ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'address_functions','exec'),ns)
rng=random.Random(20261117);checks=0
for n,k in [(4096,4096),(8192,4096),(4096,14336),(14336,4096)]:
 for transpose in [False,True]:
  physicals=[0,1,31,32,1023,1024,n*16-1,n*16,n*k-1]+[rng.randrange(n*k) for _ in range(20000)]
  for off in physicals:
   row,col=ns['inverse_bf16'](n,off,transpose)
   assert 0<=row<n and 0<=col<k
   assert ns['direct'](n,k,row,col,16,transpose)==off
   # Translate output inverse directly into source packedB4 storage; check index capacity.
   lane=(row%8)*4+(col%8)//2;i3=(row%64)//16;i4=(row//8)%2;i5=(col//8)%2
   a,b=(i5,i4) if transpose else (i4,i5)
   inner=((i3*2+a)*2+b)*2+col%2;part=inner//8;s=inner%8
   word=(col//16)*(n*2)+(row//64)*128+lane*4+part
   nib=word*8+s//2+4*(s%2)
   assert 0<=nib<n*k and nib==ns['direct'](n,k,row,col,4,transpose)
   checks+=1
report={'scope':'Independent CPU review of isolated packed BF16 candidate. No GPU/native numerical/performance claim.', 'source_hashes':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'dtype_check_ast':{'changed_top_level_statement_indices':diffs,'all_other_statements_identical':True,'original_le8_branch_preserved':True,'remaining_shape_scale_config_checks':'called unchanged by init_kernel; runtime prepare additionally restricts SM86/nonTMA/nonWS and shared capacity'},'address_checks':checks,'address_scope':'Actual-size aligned shapes; random and row/tile/expert-end boundary inverse/forward identity plus direct packedB4 address mapping. Prior literal pack-loop proof independently inspected. No padding/packed-K/interleave other than3 admitted.','native_weight_shape':'BF16 workspace has NK values; int32 view NK/2 words reshaped [1,K/16,N*8], exactly physical packedB16 storage. Single expert only.', 'arithmetic_contract':{'offsets':[246,124,120,2],'operand':'BF16(q*group_scale/4)','epilogue':'FP32 accumulator -> BF16 -> BF16 channel4 -> unchanged external global scaling. Same primitive/K partition required; no folding4 before BF16 conversion.','finite_operand_limit':'Finite E4M3 values and FP4 dyadics permit exact FP32 product prior BF16; NaN payload and hardware signed-zero behavior remain native gates.'},'memory_graph_contract':['One64MiB weight workspace; W13 conversion/GEMM completes before W2 conversion overwrites it on same capture stream.','Original and candidate source packed weight/scales updated at fixed captured pointers; geometry static.','Full scored finalizer canonical slot order retained, E1 selected slot moves across generations; invalid every-third rows generation2.','Raw observed W13 is cloned before alias reuse; activation/down/final compared separately.','Three captures allow independent allocated outputs, but pointer/poison assertions recommended to make this explicit.'],'findings':['No decisive capability/address blocker found.','Report algorithm_contract initially still says cuBLAS despite native Humming implementation; correction requested.','Comparisons assert finite and record bitwise flags; a false flag must remain held, not complete qualification.','Pipeline stage2 is restricted to native SM86 WMMA; pinned WGMMA check_config rejects fewer than3 stages.'],'next_gate':'Native scaledW13, activation, rawdown and final strict bitwise checks with changed weights/routes; no timing inference from this numerical-only pilot.'}
(P/'PACKED_BF16_INDEPENDENT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':checks,'dtype_statements_changed':diffs,'sources':len(files)}))
