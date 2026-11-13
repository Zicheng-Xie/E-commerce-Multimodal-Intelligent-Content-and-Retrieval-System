param([string]$Python = 'E:\Python 3.12\python.exe')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1"
Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python missing: $Python. Pass -Python with an installed Python 3.12+ path."
}
if (-not (Test-Path -LiteralPath '.venv312\Scripts\python.exe')) {
    uv venv --python $Python .venv312
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
if (Test-Path -LiteralPath 'requirements-retrieval.lock') {
    uv pip sync --python '.venv312\Scripts\python.exe' requirements-retrieval.lock --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
} else {
    uv pip sync --python '.venv312\Scripts\python.exe' requirements.lock
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Ready. Run .\scripts\run.ps1 doctor'
