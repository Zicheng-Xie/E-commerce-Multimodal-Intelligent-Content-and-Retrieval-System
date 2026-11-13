param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1"
Set-Location $ProjectRoot
& '.\.venv312\Scripts\python.exe' -m multimodal_ecommerce @CliArgs
exit $LASTEXITCODE

