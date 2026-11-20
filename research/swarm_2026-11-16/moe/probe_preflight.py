"""Read-only preflight shared by local MoE recovery probes; no Torch import."""
import ast
import hashlib
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = {
    'muse_spark_fp4_moe.py': '426658ee8f7e5e9b226134097d84a57b6b865d51fa132ae7ca7099857c9663f4',
    'muse_spark_nvfp4_moe.py': 'bc19ab1afd56f204b85c54ff4d80f22bb6c8214e75d978eff37cdef1eeeb0277',
}
BASELINE_SHA = '18438f083871ec322bdc409690f8b2674cef5381e1e786176eae685dddea2c0f'
REL = Path('python/sglang/srt/layers/muse_spark')

def preflight(sglang_root, require_serial=True):
    report = {'frozen_sources': {}, 'reference': 'Independent pinned stock Marlin BF16 stages with FP32 GEMM reduction; no FP32 dense oracle'}
    for name, expected in FROZEN.items():
        path = HERE / 'serving_humming_v1' / REL / name
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f'Frozen source changed: {path}: {observed}')
        report['frozen_sources'][str(path)] = observed
    path = Path(sglang_root) / REL / 'muse_spark_fp4_moe.py'
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != BASELINE_SHA:
        raise RuntimeError(f'Independent stock source changed: {path}: {observed}')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MuseSpark1xFp4Experts')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_marlin_expert_outputs')
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'moe_wna16_marlin_gemm']
    if len(calls) != 2 or not all(any(k.arg == 'use_fp32_reduce' and isinstance(k.value, ast.Constant) and k.value.value is True for k in call.keywords) for call in calls):
        raise RuntimeError('Both reference GEMMs must retain FP32 reduction')
    report['baseline_source'] = {'path': str(path), 'sha256': observed, 'fp32_reduce_gemms': len(calls)}
    report['compiler_environment'] = {k: os.environ.get(k) for k in ('MAX_JOBS', 'HUMMING_DISABLE_PARALLEL_BUILD')}
    if require_serial and any(v != '1' for v in report['compiler_environment'].values()):
        raise RuntimeError('Use launcher: MAX_JOBS=1 and HUMMING_DISABLE_PARALLEL_BUILD=1 required')
    return report
