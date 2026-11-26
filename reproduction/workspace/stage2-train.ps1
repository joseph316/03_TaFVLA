$ErrorActionPreference='Stop'
. "$PSScriptRoot/stage2-env.ps1"
$group=Join-Path $root ('runs/stage2_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Force $group | Out-Null
$group | Set-Content (Join-Path $root 'work/latest_stage2.txt')
@{status='proposal_not_team_agreed';train=@('gs_mini_obj1','gs_mini_obj3');validation='gs_mini_obj2';test='gs_mini_obj4';windows=@(1,3,5,7,10);seeds=@(42,43,44);normalization='original_cell';precision='fp32';support=10;steps=1000;probe_alphas=@(0.1,1,10,100);selection='smallest N within 5 percent of minimum validation standardized wrench RMSE; minimum 30 anchors and 2 episodes for state selection, otherwise global fallback';label_mode='fixed upstream centered pseudo-labels; W_before=2s W_after=2s, no threshold tuning on val/test';test_policy='freeze all probe coefficients, alphas and N selections before reading test outcomes'} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $group 'protocol.json')
Push-Location (Join-Path $root 'TaF-Adapter-training')
try {
 foreach($seed in @(42,43,44)) { foreach($n in @(1,3,5,7,10)) {
  $label="N$($n)_seed$seed"
  $run=Join-Path $group $label
  New-Item -ItemType Directory -Force $run | Out-Null
  & ./.venv/Scripts/python.exe -u -m script.stage2_train data.root_dir=../data/TaF-Dataset "data.window_size=$n" data.split_unit=sequence data.validation_fraction=0.33 data.stride=5 training.batch_size=16 training.num_workers=0 training.max_steps=1000 training.amp=false "training.seed=$seed" training.log_every=250 training.validate_every=1000 training.validation_batches=10 training.save_every=1000 "training.output_dir=../runs/$(Split-Path $group -Leaf)/$label" wandb.enabled=false alignment_eval.enabled=false alignment_eval.save_local_figures=false 2>&1 | Tee-Object -FilePath (Join-Path $run 'console.log')
  if($LASTEXITCODE -ne 0){throw "Training failed $label"}
 }}
} finally {Pop-Location}
