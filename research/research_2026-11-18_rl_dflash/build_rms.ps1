$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$archive = Join-Path $env:TEMP 'okc_flashinfer_0_7_2_src.tar.gz'
if (!(Test-Path -LiteralPath $archive)) {
    curl.exe -L -f -sS https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/refs/tags/v0.7.2 -o $archive
    if ($LASTEXITCODE) { throw 'FlashInfer source retrieval failed' }
}
if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne '51A02AEC7901E830FEE69DF4ED017D6BD30CBEAC55902F7B2BCE44B463539C18') { throw 'Unexpected source digest' }
$inc = Join-Path $env:TEMP 'flashinfer-0.7.2\include'
if (!(Test-Path -LiteralPath (Join-Path $inc 'flashinfer\norm.cuh'))) {
    tar -xf $archive -C $env:TEMP flashinfer-0.7.2/include
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
$vcenv = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$init = 'call "'+$vcenv+'" >nul && set'
foreach ($line in (& cmd.exe /d /c $init)) {
    if ($line -match '^([^=]+)=(.*)$') { Set-Item -LiteralPath ('Env:'+$Matches[1]) -Value $Matches[2] }
}
Push-Location $root
try {
    & nvcc.exe --shared -O3 -std=c++17 -arch=sm_86 --expt-relaxed-constexpr -I $inc '-Xcompiler=/MD' rms_bridge.cu -o rms_flashinfer_072.dll
    if ($LASTEXITCODE) {throw 'Native RMSNorm build failed'}
} finally {Pop-Location}
