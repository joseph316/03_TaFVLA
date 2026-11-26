$ErrorActionPreference='Stop'
. "$PSScriptRoot/stage2-env.ps1"
$env:OPENBLAS_NUM_THREADS='4'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$group=(Get-Content (Join-Path $root 'work/latest_stage2.txt')).Trim()
Push-Location (Join-Path $root 'TaF-Adapter-training')
try {
 & ./.venv/Scripts/python.exe -u -m script.stage2_probe $group 2>&1 | Tee-Object -FilePath (Join-Path $group 'probe.log')
 if($LASTEXITCODE -ne 0){throw 'Probe failed'}
} finally {Pop-Location}
