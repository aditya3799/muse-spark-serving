param([string]$Label,[string]$Script,[Parameter(ValueFromRemainingArguments=$true)][string[]]$ScriptArgs)
$envRoot='/home/aditya/.venvs/okc-marlin-tiles'
$base='/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/decode'
& wsl.exe -d Ubuntu -- env "CUDA_HOME=$envRoot/lib/python3.12/site-packages/nvidia/cu13" "PATH=$envRoot/bin:$envRoot/lib/python3.12/site-packages/nvidia/cu13/bin:/usr/bin:/bin" "LD_LIBRARY_PATH=$envRoot/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib" SGLANG_CACHE_DIR=/home/aditya/.cache/okc-marlin-tiles SGLANG_JIT_CACHE_DIR=/home/aditya/.cache/okc-marlin-tiles/jit "CPATH=$envRoot/python-headers/usr/include/python3.12:$envRoot/python-headers/usr/include" MAX_JOBS=1 HUMMING_DISABLE_PARALLEL_BUILD=1 OMP_NUM_THREADS=2 "$envRoot/bin/python" -u "$base/observe.py" $Label $Script @ScriptArgs
exit $LASTEXITCODE
