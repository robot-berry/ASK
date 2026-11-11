param(
    [int]$Port = 8765,
    [int]$StructuredPort = 8240,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [switch]$NoBrowser,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$structuredScript = Join-Path $PSScriptRoot "start_structured_ocr.ps1"
$webScript = Join-Path $PSScriptRoot "start_gold_review.ps1"

$structuredHealthUrl = "http://127.0.0.1:$StructuredPort/health"
$structuredHealth = $null
try {
    $structuredHealth = Invoke-RestMethod -Uri $structuredHealthUrl -TimeoutSec 3
} catch {
    # The structured service is not running yet; start it below.
}
$compatibleStructuredService = (
    $null -ne $structuredHealth -and
    $structuredHealth.ready -and
    [string]$structuredHealth.project -eq "test7-OCR" -and
    [string]$structuredHealth.service_api_version -eq "structured_service_v2" -and
    [string]$structuredHealth.engine -eq "test7-local-pp-doclayoutv2-plus-pp-formulanet" -and
    [string]$structuredHealth.warmup.profile_version -eq "synthetic_runtime_shapes_v7_scoped_vram_preclean"
)
if ($compatibleStructuredService) {
    Write-Output "结构化 OCR 服务已就绪：$structuredHealthUrl"
} else {
    & $structuredScript -Port $StructuredPort -Device $Device
}
& $webScript -Port $Port -NoBrowser -Restart:$Restart

$url = "http://127.0.0.1:$Port/ocr/"
$health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/product/health" -TimeoutSec 10
if ([string]$health.version -ne "product_ocr_v8") {
    throw "产品 OCR 接口版本不匹配：$($health.version)"
}
if (-not $NoBrowser) {
    Start-Process $url
}
Write-Output "test7-OCR 产品入口已就绪：$url"
