param(
    [int]$Port = 8240,
    [string]$MinerURoot = "",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $MinerURoot) {
    $askRoot = Split-Path -Parent (Split-Path -Parent $projectRoot)
    $MinerURoot = Join-Path $askRoot "agentpicture\MH\test1"
}
$python = Join-Path $MinerURoot ".venv\Scripts\python.exe"
$config = Join-Path $MinerURoot "mineru.json"
$healthUrl = "http://127.0.0.1:$Port/health"
$expectedProject = "test7-OCR"
$expectedApiVersion = "structured_service_v2"
$expectedEngine = "test7-local-pp-doclayoutv2-plus-pp-formulanet"
$expectedWarmupProfile = "synthetic_runtime_shapes_v7_scoped_vram_preclean"
$serviceDir = Join-Path $projectRoot "artifacts\service"
$stdout = Join-Path $serviceDir "structured-ocr.stdout.log"
$stderr = Join-Path $serviceDir "structured-ocr.stderr.log"
$pidFile = Join-Path $serviceDir "structured-ocr.pid"

$health = $null
try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
} catch {
    # The port is free or the previous process is not ready; start our service below.
}
if ($null -ne $health -and $health.ready) {
    if (
        [string]$health.project -eq $expectedProject -and
        [string]$health.service_api_version -eq $expectedApiVersion -and
        [string]$health.engine -eq $expectedEngine -and
        [string]$health.warmup.profile_version -eq $expectedWarmupProfile
    ) {
        Write-Output "结构化 OCR 服务已就绪：$healthUrl"
        exit 0
    }
    $sameProjectService = (
        [string]$health.project -eq $expectedProject -and
        [string]$health.service_api_version -eq $expectedApiVersion -and
        [string]$health.engine -eq $expectedEngine
    )
    if ($sameProjectService) {
        $listenerLine = netstat -ano -p tcp | Select-String -Pattern (
            "^\s*TCP\s+127\.0\.0\.1:" + $Port + "\s+.*\s+LISTENING\s+\d+\s*$"
        ) | Select-Object -First 1
        if (-not $listenerLine) {
            throw "检测到旧版结构化服务，但无法确定端口 $Port 的进程。"
        }
        $listenerPid = [int]((([string]$listenerLine).Trim() -split '\s+')[-1])
        $listenerProcess = Get-Process -Id $listenerPid -ErrorAction Stop
        $listenerParent = $listenerProcess.Parent
        $ownedByConfiguredRuntime = (
            $listenerProcess.ProcessName -eq 'python' -and
            $null -ne $listenerParent -and
            [System.IO.Path]::GetFullPath($listenerParent.Path) -eq
                [System.IO.Path]::GetFullPath($python)
        )
        if (-not $ownedByConfiguredRuntime) {
            throw (
                "端口 $Port 上是本项目旧接口，但进程不属于配置的 MinerU Runtime；" +
                "为避免停止其他程序，未自动替换。"
            )
        }
        Stop-Process -Id $listenerPid -Force -ErrorAction Stop
        Stop-Process -Id $listenerParent.Id -Force -ErrorAction SilentlyContinue
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            $remaining = netstat -ano -p tcp | Select-String -Pattern (
                "^\s*TCP\s+127\.0\.0\.1:" + $Port + "\s+.*\s+LISTENING\s+\d+\s*$"
            ) | Select-Object -First 1
            if (-not $remaining) {
                break
            }
            Start-Sleep -Milliseconds 100
        }
        if ($remaining) {
            throw "旧版结构化服务未完全退出，未在同一端口启动第二个实例。"
        }
        Write-Output "已停止本项目旧预热服务，正在启动 $expectedWarmupProfile。"
    } else {
        throw (
            "端口 $Port 已被旧版或其他结构化 OCR 服务占用。" +
            "当前返回 project=$($health.project), " +
            "service_api_version=$($health.service_api_version), engine=$($health.engine)；" +
            "期望 project=$expectedProject, service_api_version=$expectedApiVersion, " +
            "engine=$expectedEngine, warmup_profile=$expectedWarmupProfile。" +
            "请先停止占用进程后再启动本项目服务。"
        )
    }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "找不到 MinerU Python：$python"
}
if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "找不到 MinerU 配置：$config"
}

New-Item -ItemType Directory -Path $serviceDir -Force | Out-Null

$env:TEST7_MINERU_ROOT = $MinerURoot
$env:TEST7_FORMULA_OCR_DEVICE = $Device
$env:TEST7_MINERU_VRAM_CLEANUP = 'resident_guarded'
$arguments = @(
    "-m", "uvicorn", "test7_ocr.structured_service:app",
    "--host", "127.0.0.1", "--port", "$Port"
)
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii

$deadline = (Get-Date).AddMinutes(3)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    if ($process.HasExited) {
        throw "结构化 OCR 服务启动失败，请查看：$stderr"
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if (
            $health.ready -and
            [string]$health.project -eq $expectedProject -and
            [string]$health.service_api_version -eq $expectedApiVersion -and
            [string]$health.engine -eq $expectedEngine -and
            [string]$health.warmup.profile_version -eq $expectedWarmupProfile
        ) {
            Write-Output "结构化 OCR 服务启动成功：$healthUrl (PID $($process.Id))"
            exit 0
        }
    } catch {
        # Model initialization is still running.
    }
}
throw "结构化 OCR 服务启动超时，请查看：$stderr"
