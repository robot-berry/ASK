param(
    [Parameter(Mandatory = $true)]
    [string]$Image,
    [Parameter(Mandatory = $true)]
    [string]$SampleId,
    [ValidateSet('auto', 'hybrid', 'structured', 'rapidocr', 'sidecar', 'eval-prelabel')]
    [string]$Provider = 'auto',
    [string]$Sidecar = '',
    [int]$ServicePort = 8240,
    [switch]$NoStartService,
    [switch]$AllowEvalPrelabel
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BundledPython = Join-Path $env:HOMEDRIVE ($env:HOMEPATH.TrimStart('\') + '\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')

if (Test-Path -LiteralPath (Join-Path $ProjectRoot '.venv\Scripts\python.exe')) {
    $Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
} elseif (Test-Path -LiteralPath $BundledPython) {
    $Python = $BundledPython
} else {
    $Python = 'python'
}

$Arguments = @(
    '-m', 'test7_ocr', 'process', $Image,
    '--sample-id', $SampleId,
    '--provider', $Provider,
    '--service-url', "http://127.0.0.1:$ServicePort"
)
if ($Sidecar) {
    $Arguments += @('--sidecar', $Sidecar)
}
if ($AllowEvalPrelabel) {
    $Arguments += '--allow-eval-prelabel'
}

$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($PreviousPythonPath) { "$ProjectRoot;$PreviousPythonPath" } else { $ProjectRoot }
try {
    if (-not $NoStartService -and $Provider -in @('auto', 'hybrid', 'structured')) {
        try {
            & (Join-Path $PSScriptRoot 'start_structured_ocr.ps1') -Port $ServicePort
            if ($LASTEXITCODE -ne 0) { throw "服务启动脚本返回 $LASTEXITCODE" }
        } catch {
            if ($Provider -in @('hybrid', 'structured')) { throw }
            Write-Warning "结构化 OCR 服务不可用，将由 auto 模式降级到 RapidOCR：$($_.Exception.Message)"
        }
    }
    & $Python @Arguments
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
}
