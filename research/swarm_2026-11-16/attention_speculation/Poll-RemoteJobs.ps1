$ErrorActionPreference = 'Stop'
$outDir = Join-Path $PSScriptRoot 'remote_monitor'
[void](New-Item -ItemType Directory -Path $outDir -Force)
$stamp = Join-Path $outDir 'last_poll_epoch.txt'
$nowEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
if (Test-Path -LiteralPath $stamp) {
    $elapsed = $nowEpoch - [double](Get-Content -Raw -LiteralPath $stamp)
    if ($elapsed -lt 110) { Write-Output ('Waiting: next poll allowed in {0:N0} seconds.' -f (110-$elapsed)); exit 0 }
}
[IO.File]::WriteAllText($stamp, [string]$nowEpoch)
$config = Get-Content -Raw -LiteralPath 'D:\OpenKernelChallenge\.config\okc\config.json' | ConvertFrom-Json
$headers = @{Authorization = 'Bearer ' + $config.token}
$config = $null
$utf8 = New-Object System.Text.UTF8Encoding($false)
try {
    foreach ($job in @('9689b3114c2a45ab','fa0d2a0fcf584339','6570fbd9e1b64ce4')) {
        $queried = [DateTimeOffset]::UtcNow.ToString('o')
        $url = 'https://openkernelchallenge.dev/api/v1/jobs/' + $job
        try {
            $response = Invoke-RestMethod -Method Get -Uri $url -Headers $headers -TimeoutSec 30
            if ($response.id -ne $job) { throw 'Unexpected job ID' }
            $latest = Join-Path $outDir ('latest_' + $job + '.json')
            $previous = 'queued'
            if (Test-Path -LiteralPath $latest) { $previous = (Get-Content -Raw -LiteralPath $latest | ConvertFrom-Json).response.status }
            $snapshot = @{queried_at=$queried; endpoint=$url; http_status=200; response=$response}
            $json = $snapshot | ConvertTo-Json -Depth 80
            [IO.File]::WriteAllText($latest,$json,$utf8)
            $archive = Join-Path $outDir (([DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')) + '_' + $job + '.json')
            [IO.File]::WriteAllText($archive,$json,$utf8)
            @{job_id=$job; queried_at=$queried; status=$response.status; previous_status=$previous; transition=($previous -ne $response.status); result=$response.result} | ConvertTo-Json -Depth 80
        } catch {
            # Never serialize exception request/header objects or credentials.
            $status = $null
            if ($_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
            $failure = @{job_id=$job; queried_at=$queried; observation_error=$true; http_status=$status}
            $json = $failure | ConvertTo-Json
            [IO.File]::WriteAllText((Join-Path $outDir ('error_' + $job + '.json')),$json,$utf8)
            Write-Output $json
        }
    }
} finally {
    $headers.Clear()
    $headers = $null
}
