"""Exploratory frozen fused-phase routing vs fixed N, with episode-level diagnostics."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from script.stage3_phase_audit import prediction,KEY

ROOT=Path(__file__).resolve().parents[2];P=ROOT/'runs/stage2_20261120_115709/probes';F=ROOT/'runs/fused_phases_20261124';OUT=ROOT/'runs/fused_routing_20261124'

def main():
    OUT.mkdir(exist_ok=True)
    policy={'Loading':1,'Holding':7,'Unloading':1,'Transition / Unknown':3}
    protocol={'policy':policy,'source':'fused phase development 5-percent rule; Unknown falls back to global N=3','status':'exploratory; obj4 was already inspected in earlier analyses','label_source':'offline force-derived oracle; not deployable online','baselines':['fixed1','fixed3','fixed5','fixed7','fixed10','train_mean'],'selection_changes_after_results':False,'fusion_labels_sha256':hashlib.sha256((F/'labels.parquet').read_bytes()).hexdigest()}
    raw=json.dumps(protocol,indent=2).encode();path=OUT/'protocol.json'
    if path.exists():assert path.read_bytes()==raw
    else:path.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    labels=pd.read_parquet(F/'labels.parquet');sample=np.load(P/'N1_seed42_probe.npz');scale=sample['target_scale'];mean=sample['target_mean'];records=[];costs=[]
    for split,seq in [('dev','gs_mini_obj2'),('test','gs_mini_obj4')]:
        base=P/f'{split}_bundle';meta=pd.read_parquet(base/'metadata.parquet');all_y=np.load(base/'targets.npy');keep=(meta.sequence==seq).to_numpy();m=meta.loc[keep,KEY].merge(labels[KEY+['fused']],on=KEY,how='left',sort=False,validate='one_to_one');y=all_y[keep];assert m.fused.notna().all()
        chosen=m.fused.map(policy).to_numpy();assert np.isin(chosen,[1,3,7]).all()
        costs.append({'sequence':seq,'anchors':len(m),'mean_input_N':float(chosen.mean()),'fraction_N1':float((chosen==1).mean()),'fraction_N3':float((chosen==3).mean()),'fraction_N7':float((chosen==7).mean())})
        test=np.load(P/'test_predictions.npz') if split=='test' else None
        for seed in [42,43,44]:
            preds={}
            for n in [1,3,5,7,10]:
                key=f'N{n}_seed{seed}'
                pred=test[key][keep] if test is not None else prediction(np.load(P/f'{key}_dev_embeddings.npy')[keep],P/f'{key}_probe.npz')
                preds[f'fixed{n}']=pred
            route=preds['fixed3'].copy()
            for n in [1,7]:route[chosen==n]=preds[f'fixed{n}'][chosen==n]
            for n in [1,3,7]:assert np.array_equal(route[chosen==n],preds[f'fixed{n}'][chosen==n])
            preds['fused_route']=route;preds['train_mean']=np.broadcast_to(mean,y.shape)
            for model,pred in preds.items():
                assert np.isfinite(pred).all()
                for ep in ['ALL',*sorted(m.episode.unique())]:
                    mask=np.ones(len(m),bool) if ep=='ALL' else (m.episode==ep).to_numpy()
                    err=pred[mask]-y[mask]
                    records.append({'sequence':seq,'episode':str(ep),'seed':seed,'model':model,'anchors':int(mask.sum()),'wrench_rmse':float(np.sqrt(np.mean((err[:,:6]/scale[:6])**2))),'pressure_rmse':float(np.sqrt(np.mean(err[:,6:]**2)))})
        if test is not None:test.close()
    records=pd.DataFrame(records);records.to_csv(OUT/'metrics_per_seed_episode.csv',index=False)
    agg=records.groupby(['sequence','episode','model'],as_index=False).agg(wrench_rmse=('wrench_rmse','mean'),seed_sd=('wrench_rmse','std'),pressure_rmse=('pressure_rmse','mean'),anchors=('anchors','first'));agg.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(costs).to_csv(OUT/'input_frame_cost.csv',index=False)
    paired=[]
    for (seq,ep,seed),g in records.groupby(['sequence','episode','seed']):
        route=float(g[g.model=='fused_route'].wrench_rmse.iloc[0])
        for baseline in ['fixed3','fixed5','train_mean']:
            ref=float(g[g.model==baseline].wrench_rmse.iloc[0]);paired.append({'sequence':seq,'episode':ep,'seed':seed,'baseline':baseline,'route_minus_baseline':route-ref,'relative_percent':(route/ref-1)*100})
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'paired_differences.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    models=['train_mean','fixed1','fixed3','fixed5','fixed7','fixed10','fused_route']
    for ax,seq,title in zip(axs,['gs_mini_obj2','gs_mini_obj4'],['Development (used for N selection)','Previously seen test (exploratory)']):
        g=agg[(agg.sequence==seq)&(agg.episode=='ALL')].set_index('model').loc[models]
        ax.bar(range(len(models)),g.wrench_rmse,yerr=g.seed_sd,capsize=3,color=['#999999']+['#5a8caf']*5+['#d47a35']);ax.set_xticks(range(len(models)),models,rotation=35,ha='right');ax.set_title(title);ax.set_ylabel('Standardized wrench RMSE');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    table=[]
    for model in models:
        vals=[]
        for seq in ['gs_mini_obj2','gs_mini_obj4']:
            r=agg[(agg.sequence==seq)&(agg.episode=='ALL')&(agg.model==model)].iloc[0];vals.append(f'{r.wrench_rmse:.4f} ± {r.seed_sd:.4f}')
        table.append(f'| {model} | {vals[0]} | {vals[1]} |')
    ep_table=[]
    for ep in sorted(agg[(agg.sequence=='gs_mini_obj4')&(agg.episode!='ALL')].episode.unique()):
        g=agg[(agg.sequence=='gs_mini_obj4')&(agg.episode==ep)].set_index('model');ep_table.append(f"| {ep} | {g.loc['fixed3','wrench_rmse']:.4f} | {g.loc['fused_route','wrench_rmse']:.4f} | {g.loc['train_mean','wrench_rmse']:.4f} |")
    report='''# 결합 상태에 따른 N 선택 정책 평가

2026-11-24. 사용자와 검토한 0.2초·0.3초 OR 결합 규칙을 바탕으로 Loading=1, Holding=7, Unloading=1, Unknown=3 정책을 고정했다. Unknown에는 전체 개발 데이터 선택 N=3을 적용한다. 서로 다른 N 모델의 예측을 상태별로 골라 합친 평가이며 단일 가변 길이 모델을 새로 학습한 것이 아니다.

## 예측 오차

낮을수록 좋다. ±는 3개 seed의 표준편차이며 신뢰구간이 아니다. 개발 데이터는 N 선택에 사용되었으므로 성능 추정에 낙관 편향이 있을 수 있다. obj4도 이전에 결과를 열람했으므로 새 독립 검증이라고 부르지 않는다.

| 모델/정책 | 개발 obj2 | 기존 테스트 obj4 |
|---|---|---|
'''+ '\n'.join(table)+'''

## 테스트 에피소드별 비교

| episode | 고정 N=3 | 상태별 선택 | 학습 평균값 |
|---|---|---|---|
'''+ '\n'.join(ep_table)+'''

paired_differences.csv에서 같은 episode·seed의 정책 오차에서 기준선 오차를 뺀 값을 확인할 수 있다. 음수면 정책이 낫다. 테스트 episode가 2개뿐이므로 bootstrap 신뢰구간이나 통계적 유의성을 주장하지 않는다. 인접 frame도 독립 표본으로 취급하지 않는다.

## 계산량 해석

'''
    for c in costs:report+=f"- {c['sequence']}: 평균 선택 N={c['mean_input_N']:.3f}; N=1 {c['fraction_N1']:.1%}, N=3 {c['fraction_N3']:.1%}, N=7 {c['fraction_N7']:.1%}.\n"
    report+='''
평균 N은 처리할 입력 프레임 수의 대용치일 뿐 실측 지연시간이 아니다. 여러 모델 유지 비용과 상태 추정 비용은 포함하지 않는다. 고정 N=3보다 평균 N이 크면 정확도 이점이 있더라도 계산 절약이라고 주장할 수 없다.

## 해석 범위와 다음 단계

상태는 정답 force와 미래 구간을 참고한다. 따라서 이 결과는 상태를 이미 알고 있을 때의 분석이며 실제 온라인 동작을 입증하지 않는다. 회색을 줄이는 것, 힘 예측을 개선하는 것, 계산량을 줄이는 것은 별개의 목표다. 이 세 가지를 분리해 보고해야 한다.

비접촉 후보는 사용자 검토 기록만 유지했고 정확한 경계나 센서 baseline을 전역 적용하지 않았다. 추가 비교를 하더라도 이미 본 테스트 오차에 맞춰 N이나 분류 문턱을 바꾸지 않는다. 다음 우선순위는 여러 학습 구간의 비접촉 기준 및 분류 타당성 확인, 이후 새로운 미관측 시퀀스로 정책 검증이다. 실시간 활용을 주장하려면 미래 force 없이 상태를 추정하는 별도 평가가 필요하다.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'FUSED_ROUTING_RESULTS.md').write_text(report,encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>상태별 N 정책 평가</title><style>body{max-width:1150px;margin:40px auto;font:18px/1.7 sans-serif;padding:20px}img{width:100%}</style><h1>상태별 N 정책 평가</h1><p>Loading 1 · Holding 7 · Unloading 1 · Unknown 3. 힘 기반 오프라인 상태를 사용한 탐색적 평가입니다.</p><p><a href="RESULTS.md">전체 보고서</a> · <a href="summary.csv">에피소드별 결과</a> · <a href="paired_differences.csv">기준선 대비 차이</a></p><img src="comparison.png">',encoding='utf-8')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    hashes=json.loads((P/'freeze_hashes.json').read_text());assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    (OUT/'verification.json').write_text(json.dumps({'routing_matches_selected_model':'PASS','predictions_finite':'PASS','original_probes_unchanged':'PASS','protocol_unchanged':'PASS','model_training':False},indent=2))
    print(agg[agg.episode=='ALL'].to_string(index=False));print(pd.DataFrame(costs).to_string(index=False))

if __name__=='__main__':main()
