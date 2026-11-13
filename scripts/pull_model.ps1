# Download only after proving the active server uses the existing Llama directory.
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1"
Set-Location $ProjectRoot
& '.\.venv312\Scripts\python.exe' -m multimodal_ecommerce verify-storage
if ($LASTEXITCODE -ne 0) { throw 'Storage verification failed; download cancelled.' }
$FreeC = (Get-PSDrive C).Free
$FreeE = (Get-PSDrive E).Free
if ($FreeC -lt 70GB) { throw 'C: has less than 70 GiB free.' }
if ($FreeE -lt 8GB) { throw 'E: has less than 8 GiB free.' }
ollama pull 'qwen3.5:4b-q4_K_M'
if ($LASTEXITCODE -ne 0) { throw 'Ollama pull failed.' }
$Manifest = 'E:\LLM_for_Ollama\manifests\registry.ollama.ai\library\qwen3.5\4b-q4_K_M'
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw 'Downloaded model manifest is absent from the expected E: directory.'
}
& '.\.venv312\Scripts\python.exe' -m multimodal_ecommerce verify-storage
exit $LASTEXITCODE

