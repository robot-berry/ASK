param(
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [switch]$Restart
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw '未找到项目 .venv，请先按 README 安装依赖。'
}
$Url = "http://127.0.0.1:$Port"
$BrowserUrl = "$Url/review/?ui=34"
$existing = $null
$existingRuns = $null
$existingVersion = $null
$existingMasks = $null
$existingIntake = $null
$existingBoundaries = $null
$existingPairs = $null
try {
    $existingVersion = Invoke-RestMethod -Uri "$Url/api/version" -TimeoutSec 20
    $existing = Invoke-RestMethod -Uri "$Url/api/tasks" -TimeoutSec 20
    $existingRuns = Invoke-RestMethod -Uri "$Url/api/runs" -TimeoutSec 20
    $existingMasks = Invoke-RestMethod -Uri "$Url/api/mask-items" -TimeoutSec 20
    $existingIntake = Invoke-RestMethod -Uri "$Url/api/intake-items" -TimeoutSec 20
    $existingBoundaries = Invoke-RestMethod -Uri "$Url/api/boundary-items" -TimeoutSec 20
    $existingPairs = Invoke-RestMethod -Uri "$Url/api/pair-items" -TimeoutSec 20
} catch {
    # No compatible review service is running; start it below.
}
if (
    -not $Restart -and
    $null -ne $existing -and
    @($existing.tasks).Count -ge 1 -and
    $null -ne $existingRuns -and
    @($existingMasks.items).Count -ge 1 -and
    $null -ne $existingIntake -and
    $null -ne $existingBoundaries -and
    $null -ne $existingPairs -and
    $existingVersion.version -eq 'review_workstation_v32' -and
    $existingVersion.ui_version -eq 'review_ui_v34' -and
    $existingVersion.product_version -eq 'product_ocr_v8'
) {
    if (-not $NoBrowser) {
        Start-Process $BrowserUrl
    }
    Write-Output "人工复核服务已就绪：$Url/review/（最终 PNG 验收 + OCR Gold + 手写 Mask + 自建数据登记 + 拍照配对 + 整页题界）"
    exit 0
}

if (
    $null -ne $existingVersion -and
    [string]$existingVersion.version -like 'review_workstation_v*' -and
    $null -ne $existing -and
    @($existing.tasks).Count -ge 1
) {
    $listenerLine = netstat -ano -p tcp | Select-String -Pattern ("^\s*TCP\s+127\.0\.0\.1:" + $Port + "\s+.*\s+LISTENING\s+\d+\s*$") | Select-Object -First 1
    if (-not $listenerLine) {
        throw "检测到旧版复核服务，但无法确定其进程；请关闭占用端口 $Port 的旧服务后重试。"
    }
    $listenerParts = ([string]$listenerLine).Trim() -split '\s+'
    $oldProcessId = [int]$listenerParts[-1]
    $oldParentProcessId = $null
    try {
        $listenerProcess = Get-Process -Id $oldProcessId -ErrorAction Stop
        $listenerParent = $listenerProcess.Parent
        if (
            $null -ne $listenerParent -and
            [System.IO.Path]::GetFullPath($listenerParent.Path) -eq [System.IO.Path]::GetFullPath($Python)
        ) {
            $oldParentProcessId = $listenerParent.Id
        }
    } catch {
        # The listener itself is still the only process that must be stopped.
    }
    Stop-Process -Id $oldProcessId -Force -ErrorAction SilentlyContinue
    if ($null -ne $oldParentProcessId) {
        Stop-Process -Id $oldParentProcessId -Force -ErrorAction SilentlyContinue
    }
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $remainingListener = netstat -ano -p tcp | Select-String -Pattern ("^\s*TCP\s+127\.0\.0\.1:" + $Port + "\s+.*\s+LISTENING\s+\d+\s*$") | Select-Object -First 1
        if (-not $remainingListener) {
            break
        }
        Start-Sleep -Milliseconds 100
    }
    if ($remainingListener) {
        throw "旧版复核服务未完全退出；未在同一端口启动第二个实例。"
    }
}

$serviceProcess = Start-Process -FilePath $Python -ArgumentList @('-m', 'test7_ocr.review_server', '--port', [string]$Port) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
$ready = $false
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    Start-Sleep -Milliseconds 250
    if ($serviceProcess.HasExited) {
        throw "人工复核服务启动失败，退出码：$($serviceProcess.ExitCode)"
    }
    try {
        $startedVersion = Invoke-RestMethod -Uri "$Url/api/version" -TimeoutSec 1
        $startedMasks = Invoke-RestMethod -Uri "$Url/api/mask-items" -TimeoutSec 1
        $startedIntake = Invoke-RestMethod -Uri "$Url/api/intake-items" -TimeoutSec 1
        $startedBoundaries = Invoke-RestMethod -Uri "$Url/api/boundary-items" -TimeoutSec 1
        $startedPairs = Invoke-RestMethod -Uri "$Url/api/pair-items" -TimeoutSec 1
        if (
            $startedVersion.version -eq 'review_workstation_v32' -and
            $startedVersion.ui_version -eq 'review_ui_v34' -and
            $startedVersion.product_version -eq 'product_ocr_v8' -and
            @($startedMasks.items).Count -ge 1 -and
            $null -ne $startedIntake -and
            $null -ne $startedBoundaries -and
            $null -ne $startedPairs
        ) {
            $ready = $true
            break
        }
    } catch {
        # The local service is still starting.
    }
}
if (-not $ready) {
    $failedListenerLine = netstat -ano -p tcp | Select-String -Pattern ("^\s*TCP\s+127\.0\.0\.1:" + $Port + "\s+.*\s+LISTENING\s+\d+\s*$") | Select-Object -First 1
    if ($failedListenerLine) {
        $failedListenerId = [int]((([string]$failedListenerLine).Trim() -split '\s+')[-1])
        try {
            $failedListener = Get-Process -Id $failedListenerId -ErrorAction Stop
            if ($null -ne $failedListener.Parent -and $failedListener.Parent.Id -eq $serviceProcess.Id) {
                Stop-Process -Id $failedListenerId -Force -ErrorAction SilentlyContinue
            }
        } catch {
            # The failed child may already have exited.
        }
    }
    Stop-Process -Id $serviceProcess.Id -ErrorAction SilentlyContinue
    throw '人工复核服务未能在 30 秒内完成启动检查。'
}
if (-not $NoBrowser) {
    Start-Process $BrowserUrl
}
Write-Output "服务已启动：产品入口 $Url/ocr/；验收台 $Url/review/（PID $($serviceProcess.Id)，UI 34）"
