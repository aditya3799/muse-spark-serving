"""CPU-only fail-closed native-audit and emitted-overlay source contracts."""
import ast,copy,hashlib,json
from pathlib import Path
import native_fa3_sweep as gate
from source_extract import ROOT
P=Path(__file__).resolve().parent
checks=[]
def reject(name,contract):
    try:gate.validate_contract(contract)
    except RuntimeError:checks.append(name);return
    raise AssertionError(name+' admitted')
template=json.loads((P/'NATIVE_PRECISION_CONTRACT.template.json').read_text(encoding='utf-8-sig'))
reject('unaudited template fails before CUDA import',template)
valid=dict(template,audited=True,binary_sha256='a'*64,wrapper_sha256='b'*64,audit_evidence='CPU synthetic test only')
gate.validate_contract(valid);checks.append('syntactically complete contract accepted without native claim')
for name,key,value in [('FP16 partial rejected','output_partial_dtype','FP16'),('BF16 partial rejected','lse_partial_dtype','BF16'),('lower score accumulation rejected','score_accum_dtype','FP16'),('changed probability dtype rejected','probability_dtype','BF16'),('changed exp2 rejected','exp2_unchanged',False),('truncated key set rejected','all_valid_keys',False),('bad binary hash rejected','binary_sha256',None),('absent evidence rejected','audit_evidence',None),('wrong native revision rejected','source_revision','unknown')]:
    c=copy.deepcopy(valid);c[key]=value;reject(name,c)
# Strip ONLY the proposed support imports/decorators/tail/copy-receipts and compare
# every original AST node, including forward dtype/cast/native call bodies.
manifest=json.loads((P/'OVERLAY_MANIFEST.json').read_text())
class Strip(ast.NodeTransformer):
    def visit_ImportFrom(self,node):return None if node.module=='sglang.srt.layers.attention.fa3_split_policy' else node
    def visit_FunctionDef(self,node):
        node=self.generic_visit(node)
        node.decorator_list=[x for x in node.decorator_list if not ((isinstance(x,ast.Name) and x.id.startswith('fa3_')) or (isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id.startswith('fa3_')))]
        return node
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in ('initialize_target_fa3_split_policy','record_scheduler_copy'):return None
        return self.generic_visit(node)
for rel in manifest['sources']:
    original=ast.parse((ROOT/'sglang'/rel).read_text())
    modified=Strip().visit(ast.parse((P/'overlay'/rel).read_text()))
    assert ast.dump(original,include_attributes=False)==ast.dump(modified,include_attributes=False),rel
    checks.append('original AST preserved '+rel)
# Receipt calls must follow exact copy and zero statements, at the two pinned sites.
fa=ast.parse((P/'overlay/python/sglang/srt/layers/attention/flashattention_backend.py').read_text());sites=[]
for node in ast.walk(fa):
    for field,value in ast.iter_fields(node):
        if not isinstance(value,list):continue
        for i,stmt in enumerate(value):
            if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Name) and stmt.value.func.id=='record_scheduler_copy':
                assert ast.unparse(value[i-2])=='self._sched_meta_buf[:n] = sched'
                assert ast.unparse(value[i-1])=='self._sched_meta_buf[n:] = 0'
                sites.append(stmt.lineno)
assert len(sites)==2;checks.append('both copy receipts follow actual pinned copy/zero source')
gate.preflight();checks.append('native source preflight')
report={'status':'CPU_NATIVE_ADMISSION_AND_SOURCE_CONTRACT_PASS','count':len(checks),'checks':checks,'copy_receipt_lines':sites,'native_harness_sha256':hashlib.sha256((P/'native_fa3_sweep.py').read_bytes()).hexdigest(),'scope':'No CUDA/Torch import or native binary audit; synthetic contract positive is not runtime admission.'}
(P/'CPU_NATIVE_CONTRACT_TESTS.json').write_text(json.dumps(report,indent=2));print(report['status'],len(checks))
