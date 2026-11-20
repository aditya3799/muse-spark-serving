param(
    [switch]$Execute,
    [switch]$AuthorizedGpuSlot,
    [ValidateSet('pool','unpool','both')][string]$Geometry = 'both'
)
$ErrorActionPreference = 'Stop'
$projectionDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectionPython = '/home/aditya/.venvs/okc-marlin-tiles/bin/python'
$projectionScript = '/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/projection_communication/benchmark_small_projection_cold.py'
$projectionArgs = @('-d','Ubuntu','--','/usr/bin/timeout','--signal=TERM','--kill-after=15s','240s',$projectionPython,$projectionScript,'--gpu-authorized','--max-seconds','210','--cycles','8','--geometry',$Geometry,'--tokens','1','8','16','32','128','256')
if (-not $Execute) {
    [pscustomobject]@{status='prepared_only'; executable='wsl.exe'; arguments=$projectionArgs; windows_timeout_seconds=270; requires='Explicit root GPU handoff and AuthorizedGpuSlot switch'} | ConvertTo-Json -Depth 4
    exit 0
}
if (-not $AuthorizedGpuSlot) { throw 'Execution requires an explicitly authorized exclusive GPU slot.' }
$projectionRun = Join-Path $projectionDir ('launcher_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '_' + $PID)
New-Item -ItemType Directory -Path $projectionRun | Out-Null
$projectionLog = Join-Path $projectionRun 'stdout.txt'
$projectionErrorLog = Join-Path $projectionRun 'stderr.txt'
$projectionProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $projectionArgs -WindowStyle Hidden -RedirectStandardOutput $projectionLog -RedirectStandardError $projectionErrorLog -PassThru
# Retain a process handle before waiting; otherwise PowerShell may lose ExitCode.
$projectionProcessHandle = $projectionProcess.Handle
$projectionFinished = $projectionProcess.WaitForExit(270000)
if (-not $projectionFinished) {
    # Only this directly owned launcher is stopped. Guest timeout is independent.
    $projectionProcess.Kill()
    [pscustomobject]@{status='windows_launcher_timeout'; linux_exit_verified=$false; gpu_handoff='NOT RELEASED: coordinator must verify guest timeout/process exit'; process_id=$projectionProcess.Id} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectionRun 'launcher_result.json')
    throw 'Launcher timed out; Linux/GPU state remains unverified.'
}
$projectionExit = $projectionProcess.ExitCode
[pscustomobject]@{status='launcher_exited'; exit_code=$projectionExit; linux_exit_verified=$true; gpu_handoff='Coordinator confirms no remaining test process before next slot'; stdout=$projectionLog; stderr=$projectionErrorLog} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectionRun 'launcher_result.json')
Get-Content -LiteralPath $projectionLog
Get-Content -LiteralPath $projectionErrorLog
if ($null -eq $projectionExit) { throw 'Process exited but exit-code bookkeeping failed; inspect report and logs.' }
exit $projectionExit
