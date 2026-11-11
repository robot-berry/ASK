[CmdletBinding()]
param(
    [switch]$RequireVerified,
    [switch]$Detailed
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = $null
$profileCandidates = @(
    [Environment]::GetFolderPath('UserProfile'),
    "$env:HOMEDRIVE$env:HOMEPATH"
) | Where-Object { $_ } | Select-Object -Unique
$bundledCandidates = @(
    $profileCandidates | ForEach-Object {
        Join-Path $_ '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    }
)
$projectVenv = Join-Path $projectRoot '.venv\Scripts\python.exe'
$gateFailures = [System.Collections.Generic.List[string]]::new()

if ($env:TEST7_OCR_PYTHON -and (Test-Path -LiteralPath $env:TEST7_OCR_PYTHON)) {
    $pythonExe = $env:TEST7_OCR_PYTHON
} elseif (Test-Path -LiteralPath $projectVenv) {
    $pythonExe = $projectVenv
} else {
    foreach ($candidate in $bundledCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            $pythonExe = $candidate
            break
        }
    }
    if (-not $pythonExe) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $pythonExe = $pythonCommand.Source
        }
    }
}

if (-not $pythonExe) {
    throw '未找到 Python 3。请安装 Python 3，或把 TEST7_OCR_PYTHON 指向 python.exe。'
}

function Invoke-Test7Python {
    param([Parameter(Mandatory)][object[]]$Arguments)

    $capturedOutput = @(& $pythonExe @Arguments)
    $commandExitCode = $LASTEXITCODE
    if ($Detailed) {
        $capturedOutput | ForEach-Object { Write-Host $_ }
    }
    return [int]$commandExitCode
}

$collectionWorkspaceArgs = @((Join-Path $PSScriptRoot 'prepare_self_built_collection_workspace.py'))
if ($RequireVerified) {
    $collectionWorkspaceArgs += '--require-ready'
}
$collectionWorkspaceExit = Invoke-Test7Python -Arguments $collectionWorkspaceArgs
if ($collectionWorkspaceExit -ne 0) {
    if ($RequireVerified) {
        $gateFailures.Add('私有采集工作区尚未放齐 410 张真实图片并完成逐图隐私声明。')
    } else {
        throw '私有采集工作区结构无效；请查看 collection_readiness.json。'
    }
}

$collectionPlanArgs = @((Join-Path $PSScriptRoot 'create_self_built_collection_plan.py'), '--check')
$collectionPlanExit = Invoke-Test7Python -Arguments $collectionPlanArgs
if ($collectionPlanExit -ne 0) {
    throw '410 行自建数据采集计划缺失或已损坏。'
}

$prepareArgs = @((Join-Path $PSScriptRoot 'prepare_gaokao_physics_ocr_gold.py'))

$prepareExit = Invoke-Test7Python -Arguments $prepareArgs
if ($prepareExit -ne 0) {
    throw '自建数据队列生成失败。'
}

$maskPrepareArgs = @((Join-Path $PSScriptRoot 'prepare_handwriting_mask_eval.py'))
$maskPrepareExit = Invoke-Test7Python -Arguments $maskPrepareArgs
if ($maskPrepareExit -ne 0) {
    throw '手写像素 Mask 冻结评测队列生成失败。'
}

$validateArgs = @((Join-Path $PSScriptRoot 'prepare_gaokao_physics_ocr_gold.py'), '--check')

$validateExit = Invoke-Test7Python -Arguments $validateArgs
if ($validateExit -ne 0) {
    $gateFailures.Add('高考物理 300 题 OCR Gold 的生成事实源、图片哈希或隔离策略检查未通过。')
}

$maskValidateArgs = @((Join-Path $PSScriptRoot 'prepare_handwriting_mask_eval.py'), '--check')
if ($RequireVerified) {
    $maskValidateArgs += '--require-complete'
}
$maskValidateExit = Invoke-Test7Python -Arguments $maskValidateArgs
if ($maskValidateExit -ne 0) {
    $gateFailures.Add('手写像素 Mask 尚未完成至少 100 张双人复核，不能冻结。')
}

$boundaryArgs = @((Join-Path $PSScriptRoot 'check_question_boundary_gold.py'))
if ($RequireVerified) {
    $boundaryArgs += '--require-targets'
}
$boundaryExit = Invoke-Test7Python -Arguments $boundaryArgs
if ($boundaryExit -ne 0) {
    $gateFailures.Add('整页题界 Gold 尚未完成 60 页、300 题和单双栏双审目标，不能冻结。')
}

$pairArgs = @((Join-Path $PSScriptRoot 'check_capture_pair_gold.py'))
if ($RequireVerified) {
    $pairArgs += '--require-targets'
}
$pairExit = Invoke-Test7Python -Arguments $pairArgs
if ($pairExit -ne 0) {
    $gateFailures.Add('原稿—拍照/手写配对 Gold 尚未完成双人核验、设备和拍摄条件目标，不能冻结。')
}

$intakeArgs = @((Join-Path $PSScriptRoot 'check_self_built_intake.py'))
if ($RequireVerified) {
    $intakeArgs += '--require-targets'
}
$intakeExit = Invoke-Test7Python -Arguments $intakeArgs
if ($intakeExit -ne 0) {
    $gateFailures.Add('自建原稿、照片、手写和整页题界的采集数量或安全检查未通过。')
}

$freezeArgs = @((Join-Path $PSScriptRoot 'freeze_self_built_dataset.py'))
if ($RequireVerified) {
    $freezeArgs += '--require-targets'
}
$freezeExit = Invoke-Test7Python -Arguments $freezeArgs
if ($freezeExit -ne 0) {
    $gateFailures.Add('D0-E 冻结就绪审计未通过：仍有数量、双审、隐私、哈希、文档组或跨数据集泄漏问题。')
}

if (-not $Detailed) {
    $splitRoot = Join-Path $projectRoot 'data\splits'
    $collectionRoot = [IO.Path]::GetFullPath(
        (Join-Path $projectRoot '..\..\datasets\test7_ocr_self_built_v1\collection')
    )
    $collection = Get-Content -LiteralPath (Join-Path $collectionRoot 'collection_readiness.json') -Raw | ConvertFrom-Json
    $gold = Get-Content -LiteralPath (Join-Path $splitRoot 'gaokao_physics_ocr_gold300_summary.json') -Raw | ConvertFrom-Json
    $mask = Get-Content -LiteralPath (Join-Path $splitRoot 'testzip_handwriting_mask_eval_summary.json') -Raw | ConvertFrom-Json
    $boundary = Get-Content -LiteralPath (Join-Path $splitRoot 'question_boundary_summary.json') -Raw | ConvertFrom-Json
    $pair = Get-Content -LiteralPath (Join-Path $splitRoot 'capture_pair_summary.json') -Raw | ConvertFrom-Json
    $intake = Get-Content -LiteralPath (Join-Path $splitRoot 'self_built_intake_summary.json') -Raw | ConvertFrom-Json
    $freeze = Get-Content -LiteralPath (Join-Path $splitRoot 'self_built_freeze_readiness.json') -Raw | ConvertFrom-Json
    $nextRow = if ($collection.next_guided_row) { $collection.next_guided_row.row_id } else { '无' }
    $freezeLabel = if ($freeze.ready_to_freeze) { '已就绪' } else { '未就绪' }

    Write-Output '── test7-OCR 自建数据状态 ──'
    Write-Output ("主 OCR Gold：{0}/300（来源派生，仅用于冻结基准）" -f [int]$gold.source_generated_verified_ocr_gold_count)
    Write-Output ("真实采集：{0}/{1}；有效 {2}；隐私确认 {3}；可导入 {4}" -f [int]$collection.files_present, [int]$collection.planned_rows, [int]$collection.valid_images, [int]$collection.privacy_checked, [int]$collection.ready_batch_rows)
    Write-Output ("  原稿 {0}/{1} · 普通照片 {2}/{3} · 手写照片 {4}/{5} · 题界整页 {6}/{7}" -f [int]$collection.role_progress.clean_page.present, [int]$collection.role_progress.clean_page.target, [int]$collection.role_progress.photo_capture.present, [int]$collection.role_progress.photo_capture.target, [int]$collection.role_progress.handwriting_capture.present, [int]$collection.role_progress.handwriting_capture.target, [int]$collection.role_progress.boundary_page.present, [int]$collection.role_progress.boundary_page.target)
    Write-Output ("手写 Mask：{0}/{1} 已双审；当前候选 {2}" -f [int]$mask.verified_masks, [int]$mask.target_minimum_images, [int]$mask.selected_images)
    Write-Output ("题界 Gold：{0}/{1} 页，{2}/{3} 道题" -f [int]$boundary.verified_pages, [int]$boundary.targets.pages.minimum, [int]$boundary.verified_questions, [int]$boundary.targets.questions.minimum)
    Write-Output ("原稿配对：普通照片 {0}/{1}，手写照片 {2}/{3}" -f [int]$pair.verified_role_counts.photo_capture, [int]$pair.targets.photo_pairs.minimum, [int]$pair.verified_role_counts.handwriting_capture, [int]$pair.targets.handwriting_pairs.minimum)
    Write-Output ("冻结状态：{0}；下一计划任务：{1}" -f $freezeLabel, $nextRow)
    Write-Output ("私有采集目录：{0}" -f $collection.workspace_root)
    Write-Output '需要查看全部机器记录时使用：.\scripts\run_self_built.ps1 -Detailed'
}

if ($gateFailures.Count -gt 0) {
    $details = $gateFailures | ForEach-Object { "- $_" }
    throw ("自建数据严格门禁未通过：`n" + ($details -join "`n"))
}

if ($RequireVerified) {
    Write-Output "自建数据严格冻结门禁通过：$projectRoot"
} else {
    Write-Output "自建数据结构检查通过；严格冻结是否完成见上述状态：$projectRoot"
}
