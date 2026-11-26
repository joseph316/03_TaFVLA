# Temporal-context course study (2026-11-25)

## Provenance and scope

Base implementation: https://github.com/jo49973477/03_TaFVLA at `a8f1a17b5c2191aff35edc8c533824ec003ba964` (Youngyu Jo). This fork archives Sungwoo Kang's local experiment extensions and small result tables. Original model attribution and source history are retained. No new license is imposed on upstream work.

Data: `jiamig/taf-dataset`, revision `239c8ee8156bf11eb6d6c11c004f9a8721528d9d`. Train obj1+obj3; development obj2; initial test obj4 (two episodes). N=1,3,5,7,10, seeds 42/43/44, 1,000 actual optimizer updates each. Common ten-frame support, stride five, batch 16, FP32. Alignment uses N tactile and N force/pressure frames; downstream ridge receives tactile embeddings only and predicts the common final frame. Ridge alpha and N selection use development wrench error. Smallest N within 5% of minimum mean error is a practical selection rule, not statistical equivalence.

These are archived research scripts, not a one-command portable package. The original dataset, cached features, model checkpoints, and videos are excluded. Historical scripts have dated run paths and expect the workspace layout below. The classifier is a separate teammate-provided dependency, with no verified remote commit. Its source is not republished in this fork; request access to https://github.com/principle-of-ml-class/dataset_visualization and compare the config hash in `classifier_provenance.json`. Without that dependency, the original-label probe pipeline is incomplete. Publishing source does not make the original run fully reproducible on a clean machine.

## Workspace and environment

```text
TaF-Project/
  TaF-Adapter-training/   # clone this fork with this directory name
  dataset_visualization-main/src/taf_contact_explorer/
  dataset_visualization-main/configs/classifier.yaml
  data/TaF-Dataset/taf_dataset/gs_mini/gs_mini_obj{1,2,3}/
  data/TaF-Holdout/taf_dataset/gs_mini/gs_mini_obj4/
  tools/ffmpeg/bin/{ffmpeg.exe,ffprobe.exe}
  cache/ work/ runs/
  stage2-env.ps1 stage2-train.ps1 stage2-probe.ps1
```

Run Python modules from `TaF-Adapter-training`. Copy the three `reproduction/workspace/*.ps1` files to the parent directory; create `work` and `runs`. Tested locally: Windows, RTX 4060, Python 3.13.9, PyTorch 2.14.1+cu132. `training-requirements.txt` and `analysis-requirements.txt` are captured environment snapshots, not portable lockfiles. `uv.lock` describes the upstream package environment with filelock added; a default sync can replace CUDA torch. Choose the matching CUDA wheel source for the machine, and verify `torch.cuda.is_available()` before training. The analysis scripts additionally require pandas/scipy and the separate classifier's dependencies; FFmpeg/FFprobe must be available. Historical video export explicitly expects the Windows binaries shown above.

Download the pinned Hugging Face revision with `huggingface_hub.snapshot_download`: use patterns `taf_dataset/gs_mini/gs_mini_obj1/**` through `obj3/**` for TaF-Dataset, and `obj4/**` for TaF-Holdout. Do not mix obj4 into the training root. Sensor units are retained; physical pressure calibration is not established.

## Execution order

From the workspace parent in PowerShell:

```powershell
. ./stage2-env.ps1
Set-Location ./TaF-Adapter-training
./.venv/Scripts/python.exe -m script.stage2_features
./.venv/Scripts/python.exe -m script.stage2_labels --root ../data/TaF-Dataset --out ../work/stage2_labels_dev
Set-Location ..
./stage2-train.ps1
./stage2-probe.ps1
```

The launcher makes a new timestamped run directory and records it in `work/latest_stage2.txt`. The probe freezes choices before evaluating obj4 and refuses to overwrite an existing frozen selection. Do not remove that guard to tune on test outcomes. Stage2 feature extraction includes holdout RGB inputs but no fitting on holdout targets. The launcher uses a small training-time validation monitor; downstream probes use all development anchors.

Follow-up scripts `stage3_phase_audit.py`, `stage3_report.py`, `stage4_short_windows.py`, `stage5_review_video.py`, `stage6_fusion.py`, `stage7_routing.py`, `stage8_diagnostics.py` run in that order as Python modules. They reference the historical run `runs/stage2_20261120_115709`; for a new run, explicitly update their run constants together or reproduce in a separate workspace with that directory name. They also expect the earlier generated label, prediction, and review artifacts. `stage2_check.py` additionally depends on the earlier controlled-run checkpoint, so it is an original-run audit, not a clean-install smoke test.

The development-only convergence pilot uses N=3 seed42, max_steps=5000, validate/save_every=1000, full validation (`training.validation_batches=null`), with the same stage2 training settings. Output directory: `../runs/convergence_N3_seed42_20261124`. Run `stage8_convergence_probe.py` then `stage8_finish.py` to compare checkpoints 1000/3000/5000. Do not choose a checkpoint using test error.

## Recorded results and interpretation

| N | Development wrench RMSE | Initial test wrench RMSE |
|---|---|---|
| 1 | 0.7754 ± 0.0422 | 2.8112 ± 0.1042 |
| 3 | 0.7249 ± 0.0415 | 2.7547 ± 0.0239 |
| 5 | 0.7717 ± 0.0368 | 2.8079 ± 0.0791 |
| 7 | 0.7548 ± 0.0272 | 2.7810 ± 0.1041 |
| 10 | 0.8076 ± 0.0546 | 2.8313 ± 0.1281 |
| Training mean | 0.9153 | 2.6576 |

Errors standardize six wrench axes using training standard deviations. ± is three-seed standard deviation, not a confidence interval. N=3 leads the tested adapters but none beats the constant mean on initial test. N=10 is a valid nonuniform candidate (0.30 s first-to-last at 30 Hz); N=9 was not tested.

Offline 0.2/0.3-second label fusion selects Loading/Holding/Unloading N=1/7/1, Unknown fallback N=3. Routed development error 0.7187 vs fixed3 0.7249; previously seen test 2.7899 vs 2.7547. Mean input N increases. These later results are exploratory reanalyses, not independent holdout evidence. Labels use future force and episode-level preprocessing; routing is oracle analysis, not an online tactile-only policy. Reduced Unknown coverage does not demonstrate label accuracy. No Contact calibration and independent human labels remain unresolved. Material stiffness is not established by object IDs.

The single-seed development pilot worsens from 0.6785 at 1000 to 0.7418 at 5000 updates. This does not establish a universal convergence or overfitting conclusion. Few sequences, overlapping anchors, and distribution shifts limit generalization.

`results/` contains selected JSON/CSV summaries, not raw data/checkpoints. `result_manifest.json` records SHA256 hashes of the committed Git blobs (LF line endings; Windows working copies may use CRLF). Detailed local paths in selected text are replaced by `$TAF_PROJECT` if present. Original local artifacts remain unchanged. No full training rerun was performed for publication.

## Publication checks (2026-11-26)

Python compilation of `script/` and `dataset/` and all ten timestamp-cache unit tests passed. No full training rerun was performed for publication. This validates syntax and the cache/locking path, not every historical experiment on a clean installation.
