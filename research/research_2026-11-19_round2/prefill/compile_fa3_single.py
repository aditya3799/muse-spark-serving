"""One bounded CPU-only device compile; root grants resource slot separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('kind', choices=('paged', 'contiguous'))
args = parser.parse_args()
out = HERE / ('fa3_' + args.kind)
out.mkdir(exist_ok=True)
available_kib = int(re.search(r'^MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text(), re.M).group(1))
if available_kib < 3_500_000:
    raise RuntimeError('available host RAM below bounded compiler preflight: '+str(available_kib))
cuda = Path(sys.prefix) / 'lib/python3.12/site-packages/nvidia/cu13'
nvcc = cuda / 'bin/nvcc'
includes = [HERE/'sources/fa3/hopper', HERE/'sources/cutlass_fa3/include', cuda/'include', cuda/'include/cccl']
flags = ['-DNDEBUG', '-O3', '-std=c++17', '-DCUTE_USE_PACKED_TUPLE=1',
         '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1', '-DCUTLASS_VERSIONS_GENERATED',
         '-DCUTLASS_TEST_LEVEL=0', '-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1',
         '-DCUTLASS_DEBUG_TRACE_LEVEL=0', '-DCUTLASS_ENABLE_GDC_FOR_SM90',
         '-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED', '--expt-relaxed-constexpr',
         '--expt-extended-lambda', '--use_fast_math', '-Xcompiler=-fno-strict-aliasing',
         '-DFLASHATTENTION_DISABLE_BACKWARD', '-DFLASHATTENTION_DISABLE_DROPOUT',
         '-DFLASHATTENTION_DISABLE_UNEVEN_K', '-DFLASHATTENTION_VARLEN_ONLY',
         '-DFLASHATTENTION_DISABLE_SM8x', '-DFLASHATTENTION_DISABLE_SPARSE_MASK']
binary = out / 'kernel.cubin'
command = [str(nvcc), '--cubin', '-arch=sm_90a', *flags, '-Xptxas=-v',
           '-DDIAGNOSTIC_PAGED='+('true' if args.kind=='paged' else 'false'),
           *(f'-I{p}' for p in includes), str(HERE/'fa3_single.cu'), '-o', str(binary)]
bounded = ['/usr/bin/time', '-v', '-o', str(out/'resource.txt'),
           'prlimit', '--as=3221225472', '--core=0', '--cpu=85', '--fsize=134217728', '--', *command]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda), CUDA_PATH=str(cuda))
env['LD_LIBRARY_PATH'] = str(cuda/'lib')+':'+env.get('LD_LIBRARY_PATH','')
temp = out / 'compiler_tmp'
temp.mkdir(exist_ok=True)
env['TMPDIR'] = str(temp)
start = time.monotonic()
with (out/'compile.log').open('w') as log:
    process = subprocess.Popen(bounded, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    timeout = False
    try:
        rc = process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        timeout = True
        os.killpg(process.pid, signal.SIGKILL)
        rc = process.wait()
report = dict(kind=args.kind, returncode=rc, timeout=timeout, elapsed_s=time.monotonic()-start,
              max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              available_kib_before=available_kib, command=command,
              limits=dict(address_space_bytes=3*1024**3, seconds=90, core_bytes=0),
              flash_pin='f89bc2306632d1ec5f97b014dded4254f5b4a907',
              cutlass_pin='57e3cfb47a2d9e0d46eb6335c3dc411498efa198',
              source_sha256=hashlib.sha256((HERE/'fa3_single.cu').read_bytes()).hexdigest(),
              limitation='Offline exact specialization build only; not linked/loaded and no native FP8 runtime or H200 parity claim.')
if binary.exists() and rc == 0:
    report['cubin_bytes'] = binary.stat().st_size
    report['cubin_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    tool = cuda/'bin/cuobjdump'
    if tool.exists():
        with (out/'kernel.sass').open('w') as output:
            dump = subprocess.run([str(tool), '--dump-sass', str(binary)], stdout=output, env=env)
        report['disassembler_rc'] = dump.returncode
    sass = (out/'kernel.sass').read_text() if (out/'kernel.sass').exists() else ''
    counts = {}
    for line in sass.splitlines():
        match = re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)', line)
        if match:
            op = match.group(1)
            counts[op] = counts.get(op, 0)+1
    report['opcode_counts'] = counts
(out/'receipt.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('command','opcode_counts')}, indent=2))
if rc:
    print((out/'compile.log').read_text()[-5000:])
raise SystemExit(rc)
