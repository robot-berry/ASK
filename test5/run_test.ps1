$ErrorActionPreference = "Stop"

# 使用工程自己的虚拟环境，避免与电脑上其他 Python 项目发生版本冲突。
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$InputDir = "D:\robot\Documents\ASK\P_DATA\physic"
$OutputDir = Join-Path $ProjectDir "test_output_physic"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "未找到工程虚拟环境：$PythonExe。请先按照 README.md 安装依赖。"
}

& $PythonExe (Join-Path $ProjectDir "preprocess.py") $InputDir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "部分图片处理失败，请查看 test_output\report.csv。"
}

Write-Host "测试完成，可查看：$OutputDir\comparison_overview.jpg"
