"""Verify and package completed convergence pilot without opening heldout predictions."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.checkpoint import load_checkpoint

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/convergence_N3_seed42_20261124';OUT=RUN/'dev_probes';DIAG=ROOT/'runs/error_diagnostics_20261124'
def main():
    r=json.loads((OUT/'results.json').read_text());rows=r['results'];assert [x['step'] for x in rows]==[1000,3000,5000]
    validation=[];logs=[json.loads(s) for s in (RUN/'metrics.jsonl').read_text().splitlines()]
    for log in logs:
        if 'val/evaluated_samples' in log:validation.append({'step':log['step'],'samples':log['val/evaluated_samples']})
    assert len(validation)==5 and all(x['samples']==5408 for x in validation)
    checkpoint_info=[]
    for step in [1000,3000,5000]:
        ck=load_checkpoint(RUN/f'step_{step:08d}.pt');assert ck['step']==step
        updates={int(v['step'].item()) for v in ck['optimizer']['state'].values() if 'step' in v};assert updates=={step}
        assert ck['split']=={'train_groups':['gs_mini_obj1','gs_mini_obj3'],'validation_groups':['gs_mini_obj2']}
        checkpoint_info.append({'step':step,'optimizer_steps_verified':True});del ck
    old=json.loads((ROOT/'runs/stage2_20261120_115709/probes/frozen_selection.json').read_text())
    prior=old['validation']['N3_seed42']['metrics']['ALL']['wrench_standardized_rmse']
    delta=rows[0]['wrench_standardized_rmse']-prior
    checks={'checkpoints':checkpoint_info,'training_validation_samples':validation,'probe_validation_samples':[x['count'] for x in rows],'new_vs_original_1000_rmse_delta':delta,'test_evaluated_in_pilot':False,'single_seed':42}
    assert all(x['count']==5414 for x in rows)
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,key,title in [(axs[0],'wrench_standardized_rmse','Development wrench RMSE'),(axs[1],'pressure_raw_rmse','Development pressure RMSE')]:
        ax.plot([x['step'] for x in rows],[x[key] for x in rows],marker='o');ax.set_title(title);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2)
    fig.suptitle('N=3, seed=42; same development endpoints; lower is better');fig.tight_layout();fig.savefig(OUT/'convergence.png',dpi=150);plt.close(fig)
    table='\n'.join(f"| {x['step']:,} | {x['alpha']} | {x['wrench_standardized_rmse']:.4f} | {x['pressure_raw_rmse']:.4f} |" for x in rows)
    change=(rows[-1]['wrench_standardized_rmse']/rows[0]['wrench_standardized_rmse']-1)*100
    conclusion='감소' if change<0 else '증가'
    report=f'''# 추가 학습과 오류 진단 완료 보고

정리일 2026-11-25. N=3·seed=42의 5,000 optimizer update 학습과 1,000·3,000·5,000 체크포인트 비교를 완료했다. 모델 학습은 사용자 PC의 GPU에서 수행했다. 이번 추가 실험에서는 개발 데이터만 평가했으며 기존 테스트로 체크포인트를 선택하거나 새 성능을 보고하지 않았다.

## 동일한 촉각 힘 예측 평가

| 업데이트 | 개발에서 선택한 ridge alpha | 6축 정규화 RMSE | 압력 raw RMSE |
|---|---|---|---|
{table}

5,000회에서 1,000회 대비 개발 wrench 오차는 {abs(change):.2f}% {conclusion}했다. 이는 한 seed·한 개발 시퀀스의 결과다. 모든 N에 대한 결론이나 새로운 물체에 대한 일반화 성능으로 확대하지 않는다. 각 체크포인트의 ridge 계수는 학습 데이터로 학습하고 정규화 강도는 동일한 개발 데이터에서 선택했다. 따라서 개발 오차는 독립 테스트 추정치가 아니다.

기존 1,000회 실험의 같은 seed 개발 RMSE는 {prior:.6f}, 이번 1,000회 값은 {rows[0]['wrench_standardized_rmse']:.6f}로 차이는 {delta:.3g}다. 동일한 시작 조건 비교를 확인할 수 있다. 중간 validation 표본 수가 이전 실험보다 많다는 점은 실행 조건 차이로 남긴다.

## 완료한 검증

- 체크포인트의 실제 optimizer step이 각각 1,000/3,000/5,000인지 확인했다.
- 학습 중 validation은 1,000회마다 full batch 338개, 5,408개 표본이다. drop_last로 6개를 제외한다.
- 별도 ridge 예측 평가는 개발 anchor 전체 5,414개를 사용했다.
- 학습 obj1·obj3, 개발 obj2 분할을 유지했다. 기존 실험의 체크포인트나 분류 결과를 덮어쓰지 않았다.

## 오류 진단에서 확인한 사항

기존 테스트에서 N=3 모델의 정규화 MSE 기여는 Fy 약 41.6%, Ty 약 23.4%였다. 테스트 Fy 값의 27.7%가 학습 최솟값~최댓값 밖이고, Ty 평균은 학습 평균에서 약 -2.79 학습 표준편차 이동했다. 데이터 분포 차이가 관찰됐지만 물체 강성·접촉 위치·동작·센서 좌표계 중 원인은 아직 분리하지 못했다.

Fz도 안전한 예외는 아니다. 기존 테스트의 Fz 예측-정답 상관은 약 -0.39였다. 상황을 나누는 Fz 기준과 전체 6축 예측 성능은 구분해 해석해야 한다. 테스트를 보고 축을 제외하거나 부호를 반전하는 보정은 하지 않았다.

## 남은 일과 우선순위

자동으로 수행 가능한 이번 추가 학습·평가·진단·검증은 완료했다. 사람 정답에 의한 분류 정확도, 비접촉 기준의 여러 구간 검증, 새로운 미관측 시퀀스에서의 일반화 평가, 온라인 상황 추정은 아직 미완료다.

다음 실험은 현재 결과를 바탕으로 학습 길이와 데이터 분할을 사전에 고정하고 추가 seed 및 새로운 시퀀스를 확보하는 것이다. 모델 학습 손실만 줄었다는 이유로 전체 N 실험을 확대하지 않는다. 원래 분류를 계속 바꿔 이미 본 테스트 성능에 맞추는 것도 피한다.

축별 CSV와 그래프는 ../error_diagnostics_20261124, 이번 체크포인트·학습 로그는 이 실험 폴더에 있다. 재실행 코드는 script/stage8_diagnostics.py, stage8_convergence_probe.py, stage8_finish.py다.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'LATEST_RESULTS.md').write_text(report,encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>추가 학습 결과</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:18px/1.7 sans-serif}img{width:100%}</style><h1>추가 학습과 오류 진단</h1><p>N=3, seed=42의 개발 데이터 비교입니다. 테스트 추가 평가와 모델 선택은 수행하지 않았습니다.</p><p><a href="RESULTS.md">완료 보고서</a> · <a href="summary.csv">평가 수치</a> · <a href="../../error_diagnostics_20261124/RESULTS.md">오류 진단</a></p><img src="convergence.png"><img src="../../error_diagnostics_20261124/axis_errors.png">',encoding='utf-8')
    print(json.dumps({'rows':rows,'change_percent':change,'checks':checks},indent=2))
if __name__=='__main__':main()
