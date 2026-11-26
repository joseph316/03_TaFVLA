"""Package the fixed phase sensitivity audit and local review index."""
from pathlib import Path
import json, html
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];out=ROOT/'runs/phase_audit_20261124';p=ROOT/'runs/stage2_20261120_115709/probes'
a=pd.read_csv(out/'comparison.csv');counts=pd.read_csv(out/'phase_counts.csv');sel=pd.read_csv(out/'development_selection.csv');review=pd.read_csv(out/'human_review.csv',keep_default_na=False)
old=json.loads((p/'results.json').read_text());cfg=json.loads((out/'calibration.json').read_text())
for variant in a.variant.unique():
    for n in [1,3,5,7,10]:
        row=a[(a.variant==variant)&(a.sequence=='gs_mini_obj4')&(a.phase=='ALL')&(a.N==n)].iloc[0]
        assert np.isclose(row.rmse,old['test_aggregate'][str(n)]['ALL']['wrench_standardized_rmse_mean'],rtol=1e-6)
assert len(list((out/'review').glob('*.png')))==len(review)
table=[]
for phase in ['Loading','Holding','Unloading']:
    vals=[int(sel[(sel.variant==v)&(sel.phase==phase)].N_within_5_percent.iloc[0]) for v in ['original_phase','noise_1s','noise_2s']]
    table.append(f'| {phase} | {vals[0]} | {vals[1]} | {vals[2]} |')
counttable=[]
for v in ['original_phase','noise_1s','noise_2s']:
    g=counts[(counts.sequence=='gs_mini_obj2')&(counts.variant==v)]
    numbers=[int(g[g.phase==ph].frames.sum()) for ph in ['Loading','Holding','Unloading','Transition / Unknown']]
    counttable.append('| '+v+' | '+' | '.join(map(str,numbers))+' |')
threshold=max(cfg['noise_sigma']*6,cfg['force_iqr']*.02)
# Korean project documentation, preserving original experiment artifacts.
body=f'''# 상황 분류 보정 및 N 비교 결과

2026-11-24. 기존 분류를 보존하고 노이즈를 고려한 1초·2초 오프라인 분류를 추가했다. 15개 모델을 재학습하지 않고 기존 예측을 새 구간으로 나눠 집계했다. 보정 규칙이 사람의 정답보다 정확한지는 아직 검증하지 않았으며, 아래 선택 차이는 분류 기준에 대한 민감도 결과다.

## 변경 내용

기존 방식은 구간 단위 규칙, 새 방식은 고정 길이 중심 윈도우에서 힘의 회귀 기울기와 양 끝 변화량을 함께 확인한다. 따라서 시간 구간 방식과 임계값을 함께 바꾼 비교이며, 어느 한 요소의 효과만 분리한 실험은 아니다. 1초를 주 설정, 2초를 민감도 설정으로 결과 열람 전에 지정했다.

obj1·obj3의 기존 전처리 힘에서 2차 차분의 MAD로 구한 노이즈 대용치는 {cfg['noise_sigma']:.5f}, 힘 IQR은 {cfg['force_iqr']:.5f}다. 변화량 기준은 max(6×노이즈, 0.02×IQR)={threshold:.5f}이다. Loading/Unloading은 이 변화량과 구간 길이로 나눈 기울기 기준을 함께 넘겨야 한다. Holding은 두 기준의 절반 이내로 제한했다. 0.4초 미만 구간과 경계는 Unknown이다. 이 수치는 경험적 설정이며 물리적으로 검증된 센서 오차가 아니다.

상수 baseline 이동은 기울기와 변화량을 바꾸지 않는다. 기존 episode별 자동 부호·baseline 전처리는 유지했다. 이번 작업으로 부호나 baseline 자체가 검증됐다는 의미는 아니다. 물체별 신호 범위 정규화도 수행하지 않았다.

## 개발 데이터에서 선택되는 N

| 상황 | 기존 분류 | 보정 1초 | 보정 2초 |
|---|---|---|---|
{chr(10).join(table)}

선택 규칙은 3개 seed 평균 최소 오차에서 5% 이내인 가장 작은 N이다. 즉 N=5 선택이 반드시 오차 최솟값 N=5라는 뜻은 아니다. 예를 들어 1초 Holding의 최소 평균 오차는 N=7에 있지만 N=5도 5% 범위에 든다. 전체 데이터의 선택은 세 분류 모두 N=3이다. 분류만 바꿨으므로 전체 오차는 변하지 않으며 이 불변성을 검증했다.

개발 시퀀스 obj2의 원본 프레임 분포:

| 분류 | Loading | Holding | Unloading | Unknown |
|---|---|---|---|---|
{chr(10).join(counttable)}

기존과 보정 1초 라벨의 일치율은 obj1 59.6%, obj2 58.5%, obj3 63.5%, obj4 50.6%다. 일치율은 정확도가 아니다. 분류마다 평가에 포함되는 프레임이 다르므로, 특정 상황의 오차가 낮아졌다는 이유만으로 분류기가 개선됐다고 판단할 수 없다.

## 접촉 여부와 물체 차이

No Contact의 기준값으로 확인된 구간이 없으므로 새 분류는 contact_status=Unverified를 명시한다. Loading/Holding/Unloading은 현재 힘 변화 단계이며 접촉 사실을 보증하지 않는다. 낮은 압력 구간을 임의로 비접촉으로 선언하거나 pressure threshold를 결과에 맞춰 조정하지 않았다. 현재 단계는 완성된 접촉 분류기가 아닌 비교용 후보 규칙이다.

object_phase_N.png와 comparison.csv에 물체별·상황별 결과를 분리했다. obj1·obj3은 모델과 probe가 학습한 데이터라 진단용이며, obj2는 개발 데이터, obj4는 이미 열람한 테스트의 탐색적 재분석이다. 물체별 차이를 곧바로 강성 차이라고 해석할 수 없다. 직접적인 누름 변위나 강성 정답이 없는 현재 자료로는 형상·동작·재료 효과를 분리하지 못한다.

## 실제 구간 검토 자료

학습 시퀀스에서 {len(review)}개 검토 후보를 골라 review/에 각 후보의 전후 0.5초 촉각 이미지 3장, 압력 지도, 주변 힘 곡선, 기존·1초·2초 라벨을 함께 저장했다. index.html에서 목록을 열 수 있다. human_review.csv의 human_phase와 human_contact는 미기입 상태다. Loading 등의 명확한 긴 구간 위주 후보이므로 이것만으로 전체 분류 정확도를 추정하면 안 된다.

육안 표본 확인에서 00번은 이미지·압력 분포가 비슷하고 힘도 비교적 일정한 유지 구간이었다. 02번은 힘이 증가하는 구간이지만 앞뒤 경계는 설정별로 달랐다. 08번은 압력이 낮고 뚜렷한 접촉 변형이 약한 프레임 뒤에 이미지·압력 변화가 나타나는 후보였다. 08번은 비접촉 기준 검토에 유용하지만 이미지 세 장만으로 비접촉 정답을 확정하지 않았다. 이 관찰은 사람 검수 결과나 정확도 측정을 대신하지 않는다.

## 검증과 다음 순서

단조 증가·감소·상수 신호 분류, 상수 offset 불변성, N별 동일 시점 매칭, 예측 유한값, 원본 probe/선택 파일 해시 유지 및 전체 테스트 오차 불변성을 검증했다. 영상 캐시는 30개 episode의 무작위 윈도우에서 원래 디코더와 픽셀 및 PTS 일치를 확인했다. 학습·모델 가중치 변경은 없다.

다음에는 우선 검토 후보에서 접촉 시작 전 구간과 안정 접촉 구간을 확인해 센서 기준값을 정해야 한다. 이후 분류 경계·불일치 구간을 포함한 별도 검토 표본으로 규칙을 검증한다. 그 뒤 기준을 고정하고 새로운 미관측 시퀀스에서 재검증한다. 현재 결과만으로 보정안을 기존 분류의 정식 대체로 채택하지 않는다.

재실행 코드는 TaF-Adapter-training/script/stage3_phase_audit.py와 stage3_report.py다. stage2-env.ps1을 적용한 같은 가상환경에서 모듈로 실행한다. 기존 calibration.json과 동일한 설정일 때만 재실행하며 규칙을 바꾸려면 별도 audit 폴더를 사용한다.
'''
(out/'RESULTS.md').write_text(body,encoding='utf-8');(ROOT/'PHASE_AUDIT_RESULTS.md').write_text(body,encoding='utf-8')
items=[]
for i,row in review.iterrows():
    label=f'{i:02d} | {row.sequence} episode {row.episode} frame {row.frame} | {row.suggested_phase}'
    items.append(f'<li><a href="review/{i:02d}.png">{html.escape(label)}</a></li>')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>상황 분류 검토 자료</title><style>body{max-width:1000px;margin:40px auto;font:17px/1.7 sans-serif;padding:20px}li{margin:10px 0}img{width:100%}a{color:#1359a0}</style><h1>상황 분류 검토 자료</h1><p>원래 분류와 노이즈 보정 1초·2초 분류를 비교합니다. 접촉 여부는 미확인입니다. 아래 후보의 라벨은 자동 제안이며 정답이 아닙니다.</p><p>색상: 주황 Loading · 녹색 Holding · 보라 Unloading · 회색 Unknown.</p><p><a href="RESULTS.md">보고서</a> · <a href="human_review.csv">검토 CSV</a> · <a href="comparison.csv">물체별 N 결과 CSV</a></p><h2>검토 순서</h2><p>00번 유지 구간, 02번 증가 구간, 08번 낮은 압력 후보부터 확인하세요. CSV의 human_phase, human_contact, notes에 판정을 기록할 수 있습니다. 불확실하면 Unknown으로 남깁니다.</p><ol>'''+''.join(items)+'''</ol><h2>물체별 N 비교</h2><p>obj1·3은 학습 데이터의 진단 결과입니다. 오차 막대는 seed 표준편차이며 신뢰구간이 아닙니다.</p><img src="object_phase_N.png" alt="물체별 상황별 N 결과"></html>'''
(out/'index.html').write_text(page,encoding='utf-8')
v=json.loads((out/'verification.json').read_text());v.update({'all_N_overall_error_invariant':True,'review_sheet_count':len(review)});(out/'verification.json').write_text(json.dumps(v,indent=2))
print(json.dumps({'report':str(out/'RESULTS.md'),'review_sheets':len(review),'checks':v},ensure_ascii=False,indent=2))
