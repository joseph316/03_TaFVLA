"""User-proposed OR fusion, with explicit opposing-direction conflicts."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from script.stage3_phase_audit import prediction,KEY,COLORS

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/short_windows_20261124';P=ROOT/'runs/stage2_20261120_115709/probes';OUT=ROOT/'runs/fused_phases_20261124'
L='Loading';U='Unloading';H='Holding';X='Transition / Unknown'

def fuse(a,b):
    loading=(a==L)|(b==L);unloading=(a==U)|(b==U);conflict=loading&unloading
    result=np.full(len(a),X,object);result[loading&~unloading]=L;result[unloading&~loading]=U;result[(a==H)&(b==H)]=H
    return result,conflict

def main():
    OUT.mkdir(exist_ok=True)
    # Exhaustive symmetric truth-table verification, including opposite directions.
    truth=[]
    for a in [L,U,H,X]:
        for b in [L,U,H,X]:
            r,c=fuse(np.array([a]),np.array([b]));reverse,_=fuse(np.array([b]),np.array([a]));assert r[0]==reverse[0]
            expected=X if {a,b}=={L,U} else L if L in (a,b) else U if U in (a,b) else H if a==b==H else X
            assert r[0]==expected;truth.append({'phase_02':a,'phase_03':b,'fused':r[0],'conflict':bool(c[0])})
    pd.DataFrame(truth).to_csv(OUT/'truth_table.csv',index=False)
    d=pd.read_parquet(BASE/'labels.parquet');d['fused'],d['direction_conflict']=fuse(d['w0.2'].to_numpy(),d['w0.3'].to_numpy())
    d['contact_status']='Unverified';d.to_parquet(OUT/'labels.parquet',index=False)
    notes={'source':'user review in conversation, 2026-11-24','clip':'08_contact_transition','sequence':'gs_mini_obj1','episode':0,'contact_assessment':'likely no contact before approximately 2.1 seconds','boundary_uncertain':True,'not_applied_as_global_contact_threshold':True,'phase_definition':'Record force decrease as Unloading, no extra impact label','fusion':'OR Loading/Unloading; opposite directions Unknown; Holding only when both Holding; other Unknown'}
    (OUT/'user_review.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=[]
    for seq,g in d.groupby('sequence'):
        row={'sequence':seq,'frames':len(g),'conflict_frames':int(g.direction_conflict.sum()),'unknown_02':float((g['w0.2']==X).mean()),'unknown_03':float((g['w0.3']==X).mean()),'unknown_fused':float((g.fused==X).mean())}
        for v in ['w0.2','w0.3','fused']:
            count=sum(int((ep[v].to_numpy()[1:]!=ep[v].to_numpy()[:-1]).sum()) for _,ep in g.groupby('episode'));row[v+'_switches_per_min']=count/(len(g)/30)*60
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'coverage.csv',index=False)
    metrics=[];probe=np.load(P/'N1_seed42_probe.npz');scale=probe['target_scale'];mean=probe['target_mean']
    for split in ['dev','test']:
        folder=P/f'{split}_bundle';meta=pd.read_parquet(folder/'metadata.parquet');y=np.load(folder/'targets.npy');m=meta[KEY].merge(d[KEY+['fused']],on=KEY,how='left',sort=False,validate='one_to_one');assert m.fused.notna().all()
        preds={}
        if split=='dev':
            for n in [1,3,5,7,10]:
                for s in [42,43,44]:
                    k=f'N{n}_seed{s}';preds[k]=prediction(np.load(P/f'{k}_dev_embeddings.npy'),P/f'{k}_probe.npz')
        else:
            with np.load(P/'test_predictions.npz') as z:
                assert np.array_equal(z['targets'],y)
                for n in [1,3,5,7,10]:
                    for s in [42,43,44]:preds[f'N{n}_seed{s}']=z[f'N{n}_seed{s}']
        preds['train_mean']=np.broadcast_to(mean,y.shape)
        for key,pred in preds.items():
            assert np.isfinite(pred).all();error=((pred[:,:6]-y[:,:6])/scale[:6])**2
            for seq in m.sequence.unique():
                for phase in ['ALL',L,H,U,X]:
                    mask=(m.sequence==seq).to_numpy().copy()
                    if phase!='ALL':mask&=(m.fused==phase).to_numpy()
                    if mask.any():metrics.append({'sequence':seq,'phase':phase,'model':key,'anchors':int(mask.sum()),'episodes':int(m.loc[mask,'episode'].nunique()),'rmse':float(np.sqrt(error[mask].mean()))})
    metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'metrics.csv',index=False);a=metrics[metrics.model!='train_mean'].copy();a['N']=a.model.str.extract(r'N(\d+)').astype(int)
    agg=a.groupby(['sequence','phase','N'],as_index=False).agg(rmse=('rmse','mean'),seed_sd=('rmse','std'),anchors=('anchors','first'),episodes=('episodes','first'));agg.to_csv(OUT/'comparison.csv',index=False)
    selected=[]
    for phase,g in agg[agg.sequence=='gs_mini_obj2'].groupby('phase'):
        n=int(g[g.rmse<=g.rmse.min()*1.05].N.min()) if g.anchors.iloc[0]>=30 and g.episodes.iloc[0]>=2 else None
        selected.append({'phase':phase,'selected_N':n,'minimum_error_N':int(g.loc[g.rmse.idxmin(),'N']),'anchors':int(g.anchors.iloc[0])})
    pd.DataFrame(selected).to_csv(OUT/'development_selection.csv',index=False)
    manifest=json.loads((ROOT/'runs/video_review_20261124/manifest.json').read_text());cards=[]
    for item in manifest:
        g=d[(d.sequence==item['sequence'])&(d.episode==item['episode'])&(d.frame>=item['start_frame'])&(d.frame<=item['end_frame'])]
        fig,axs=plt.subplots(2,1,figsize=(12,6),sharex=True,gridspec_kw={'height_ratios':[2,1]});axs[0].plot(g.timestamp,g.signed_force,lw=1);axs[0].set_title(item['id']+' — contact unverified; phase = force trend');axs[0].set_ylabel('Signed RAW Fz')
        for j,v in enumerate(['w0.2','w0.3','fused']):
            for phase,color in COLORS.items():
                mask=g[v]==phase;axs[1].scatter(g.loc[mask,'timestamp'],np.full(mask.sum(),j),c=color,s=12,label=phase if j==0 else None)
        conflict=g.direction_conflict
        axs[1].scatter(g.loc[conflict,'timestamp'],np.full(conflict.sum(),2),c='black',marker='x',s=25,label='Opposing directions')
        axs[1].set_yticks([0,1,2],['0.2s','0.3s','Fused']);axs[1].set_xlabel('Episode timestamp (s)');axs[1].legend(ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.3));fig.tight_layout();fig.savefig(OUT/(item['id']+'.png'),dpi=130);plt.close(fig)
        cards.append(f'<h2>{item["id"]}</h2><img src="{item["id"]}.png"><video controls preload="metadata" src="../video_review_20261124/{item["id"]}.mp4"></video><p>위 그림은 새 결합 판정, 아래 영상은 기존 판정이 표시된 동기화 원본 검토 영상입니다.</p>')
    report='''# 0.2초와 0.3초 판정 결합

사용자 검토에 따라 힘 감소는 기존 Unloading으로 기록하며 충격 라벨을 추가하지 않았다. 08번 약 2.1초 이전의 비접촉 가능성은 user_review.json에 경계가 불확실한 검토 의견으로 저장했다. 창 판정은 현재 힘 변화이며 접촉 여부를 보증하지 않는다. 비접촉 의견을 라벨 전체에 자동 적용하지 않았다.

두 설정 중 어느 하나라도 Loading이면 Loading, 어느 하나라도 Unloading이면 Unloading이다. 단 두 방향이 충돌하면 Unknown으로 둔다. 둘 다 Holding일 때만 Holding이며 Holding+Unknown은 Unknown으로 둔다. 후자는 새로운 증가·감소 증거가 없는 경우를 임의로 확정하지 않기 위한 처리다. truth_table.csv에 16개 조합이 있다.

## 회색 구간 변화

| 시퀀스 | 0.2초 Unknown | 0.3초 Unknown | 결합 Unknown | 방향 충돌 프레임 |
|---|---|---|---|---|
'''
    for r in summary:report+=f"| {r['sequence']} | {r['unknown_02']:.1%} | {r['unknown_03']:.1%} | {r['unknown_fused']:.1%} | {r['conflict_frames']} |\n"
    report+='\n## 개발 데이터 N 선택\n\n| 상황 | 선택 N | 최소 오차 N | anchor 수 |\n|---|---|---|---|\n'
    for r in selected:report+=f"| {r['phase']} | {r['selected_N']} | {r['minimum_error_N']} | {r['anchors']} |\n"
    report+='''
선택 N은 최소 평균 오차의 5% 이내인 가장 작은 N이다. 회색 구간이 줄어드는 것은 분류 정확도가 개선됐다는 증거가 아니다. OR 규칙은 한쪽의 잘못된 증가·감소도 채택한다. coverage.csv의 라벨 전환 횟수와 검토 그림을 함께 확인해야 한다. 충돌 구간을 검은 ×로 표시했다.

모델 재학습 없이 기존 예측을 재집계했다. obj1·3은 학습 진단, obj2는 개발, 이미 확인한 obj4는 탐색적 재분석이다. 기존 가중치와 probe를 유지했다. 16가지 조합·순서 대칭성·예측 유한값·전체 테스트 오차 불변성을 검사했다. 사람 정답 대비 정확도는 측정하지 않았다.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>0.2초와 0.3초 결합</title><style>body{max-width:1200px;margin:30px auto;font:17px/1.7 sans-serif;padding:20px}img,video{width:100%}</style><h1>0.2초 + 0.3초 결합 판정</h1><p>주황: 증가 · 녹색: 유지 · 보라: 감소 · 회색: 미확정 · 검은 ×: 증가/감소 충돌</p><a href="RESULTS.md">결과 보고서</a>'+''.join(cards),encoding='utf-8')
    prev=json.loads((P/'results.json').read_text())
    for n in [1,3,5,7,10]:
        r=agg[(agg.sequence=='gs_mini_obj4')&(agg.phase=='ALL')&(agg.N==n)].iloc[0];assert np.isclose(r.rmse,prev['test_aggregate'][str(n)]['ALL']['wrench_standardized_rmse_mean'],rtol=1e-6)
    hashes=json.loads((P/'freeze_hashes.json').read_text());assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    (OUT/'verification.json').write_text(json.dumps({'truth_table_16_cases':'PASS','symmetry':'PASS','original_hashes':'PASS','overall_error_invariance':'PASS','human_accuracy':'not measured'},indent=2))
    print(json.dumps({'coverage':summary,'selection':selected},indent=2))

if __name__=='__main__':main()
