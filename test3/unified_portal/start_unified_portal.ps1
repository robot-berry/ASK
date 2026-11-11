param(
    [int]$Port = $(if ($env:TEST3_UNIFIED_PORT) { [int]$env:TEST3_UNIFIED_PORT } else { 8030 }),
    [string]$HostAddress = $(if ($env:TEST3_UNIFIED_HOST) { $env:TEST3_UNIFIED_HOST } else { "0.0.0.0" })
)

$ErrorActionPreference = "Stop"
if ($Port -in @(8010, 8020)) {
    throw "The unified portal must not use the existing physics/math ports 8010 or 8020."
}

$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Candidates = @(
    (Join-Path $Project ".venv\Scripts\python.exe"),
    (Join-Path $Project "..\..\test1\.venv\Scripts\python.exe")
)
$Python = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Python) {
    throw "Python environment not found. Install FastAPI and uvicorn first."
}

$Source = Join-Path $Project "source"
$env:PYTHONPATH = $Source
& $Python -m uvicorn unified_portal_app:app --app-dir $Source --host $HostAddress --port $Port

