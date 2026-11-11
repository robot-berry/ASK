$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project ".venv\Scripts\python.exe"
$stdout = Join-Path $project "server.stdout.log"
$stderr = Join-Path $project "server.stderr.log"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found: $python"
}

$alreadyRunning = netstat -ano | Select-String ':8000\s+.*LISTENING'
if ($alreadyRunning) {
    Write-Host "Question image service is already running on port 8000."
    exit 0
}

# Some Codex/PowerShell hosts expose both `Path` and `PATH`. Start-Process
# builds a case-insensitive environment dictionary and otherwise fails before
# launching Python. Keep the canonical Windows `PATH` entry only.
$pathKeys = @([System.Environment]::GetEnvironmentVariables("Process").Keys |
    Where-Object { $_.ToString().ToUpperInvariant() -eq "PATH" })
if ($pathKeys.Count -gt 1 -and ($pathKeys -ccontains "Path") -and ($pathKeys -ccontains "PATH")) {
    [System.Environment]::SetEnvironmentVariable("Path", $null, "Process")
}

Start-Process `
    -FilePath $python `
    -ArgumentList @(
        "-m", "uvicorn", "app.main:app",
        "--app-dir", $project,
        "--host", "0.0.0.0",
        "--port", "8000"
    ) `
    -WorkingDirectory $project `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

Start-Sleep -Seconds 2
$health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 8
if ($health.status -ne "ok") {
    throw "Question image service failed its health check."
}

$wifi = Get-NetIPConfiguration -ErrorAction SilentlyContinue |
    Where-Object { $_.IPv4DefaultGateway -and $_.InterfaceAlias -match "WLAN|Wi-Fi" } |
    Select-Object -First 1
$address = if ($wifi) { $wifi.IPv4Address.IPAddress } else { "computer-ip" }
Write-Host "Question image service started: http://${address}:8000"
