# Dot-source this file to use only project-owned caches.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
$env:UV_CACHE_DIR = Join-Path $ProjectRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot '.runtime\python'
$env:UV_PYTHON_CACHE_DIR = Join-Path $ProjectRoot '.cache\python'
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
$env:HF_HOME = Join-Path $ProjectRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $ProjectRoot '.cache\torch'
$env:XDG_CACHE_HOME = Join-Path $ProjectRoot '.cache'
$env:PYLINTHOME = Join-Path $ProjectRoot '.cache\pylint'
$env:TEMP = Join-Path $ProjectRoot 'tmp'
$env:TMP = $env:TEMP
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = 'E:\LLM_for_Ollama'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
