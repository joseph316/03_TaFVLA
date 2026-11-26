"""Fixed short-window sensitivity analysis, not classifier accuracy validation."""
from pathlib import Path
import hashlib,json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from script.stage3_phase_audit import prediction,robust_sigma,KEY,PHASES,COLORS

ROOT=Path(__file__).resolve().parents[2];P=ROOT/'runs/stage2_20261120_115709/probes'
PREV=ROOT/'runs/phase_audit_20261124';OUT=ROOT/'runs/short_windows_20261124'
WS=[.1,.2,.3,.4,.5];NS=[1,3,5,7,10];SEEDS=[42,43,44]

def classify(t,f,seconds,cfg):
    dt=float(np.median(np.diff(t)));intervals=int(round(seconds/dt));points=intervals+1
    left=intervals//2;right=intervals-left
    windows=np.lib.stride_tricks.sliding_window_view(f,points)
    x=np.arange(points)*dt;x-=x.mean();duration=intervals*dt
    slopes=windows@x/(x@x);changes=windows[:,-1]-windows[:,0]
    floor=cfg['range_fraction']*cfg['force_iqr'];noise=cfg['noise_sigma']
    change_thr=max(3*np.sqrt(2)*noise,floor)
    slope_thr=max(3*noise/np.sqrt(x@x),floor/duration)
    inner=np.full(len(windows),'Transition / Unknown',object)
    inner[(slopes>=slope_thr)&(changes>=change_thr)]='Loading'
    inner[(slopes<=-slope_thr)&(changes<=-change_thr)]='Unloading'
    inner[(abs(slopes)<=slope_thr*.5)&(abs(changes)<=change_thr*.5)]='Holding'
    labels=np.full(len(f),'Transition / Unknown',object);labels[left:len(f)-right]=inner
    cuts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if b-a<cfg['min_run_frames']:labels[a:b]='Transition / Unknown'
    return labels,{'requested_seconds':seconds,'points':points,'actual_span_seconds':duration,'left_frames':left,'right_frames':right,'center_offset_seconds':(right-left)*dt/2,'change_threshold':change_thr,'slope_threshold':slope_thr}

def raw_table():
    audit=json.loads((ROOT/'work/stage2_labels_dev/audit.json').read_text())['episodes']+json.loads((P/'labels_test/audit.json').read_text())['episodes']
    calibration={(x['sequence'],x['episode']):x for x in audit};parts=[]
    for obj in range(1,5):
        base=ROOT/('data/TaF-Holdout' if obj==4 else 'data/TaF-Dataset')/'taf_dataset/gs_mini'/f'gs_mini_obj{obj}'
        for file in sorted((base/'data').rglob('*.parquet')):
            d=pd.read_parquet(file);force=np.stack(d['observation.force_torque'])[:,2]
            g=pd.DataFrame({'sequence':f'gs_mini_obj{obj}','episode':d.episode_index,'frame':d.frame_index,'timestamp':d.timestamp,'raw_fz':force})
            for ep,part in g.groupby('episode'):
                c=calibration[(f'gs_mini_obj{obj}',ep)];part=part.copy();part['signed_force']=part.raw_fz*c['auto_sign']-c['episode_baseline'];parts.append(part)
    result=pd.concat(parts,ignore_index=True);assert not result.duplicated(KEY).any();return result

def main():
    OUT.mkdir(exist_ok=True);d=raw_table();train=d[d.sequence.isin(['gs_mini_obj1','gs_mini_obj3'])]
    noise=[robust_sigma(np.diff(g.signed_force.to_numpy(),n=2))/np.sqrt(6) for _,g in train.groupby(['sequence','episode'])]
    cfg={'fit_sequences':['gs_mini_obj1','gs_mini_obj3'],'noise_sigma':float(np.median(noise)),'force_iqr':float(train.signed_force.quantile(.75)-train.signed_force.quantile(.25)),'range_fraction':.02,'min_run_frames':2,'windows':WS,'noise_estimator':'median episode MAD of RAW signed force second differences / sqrt(6); proxy, motion can contaminate','smoothing':'none; local regression only','contact_status':'Unverified','inherited':'stage2 episode sign and baseline, offline','selection':'no winning classification window chosen from model errors','test_role':'previously inspected obj4, exploratory only'}
    # Synthetic known-phase and temporal-support checks before evaluating model errors.
    t=np.arange(301)/30;gain=max(100*cfg['noise_sigma'],10*cfg['force_iqr'],10)
    for w in WS:
        for f,label in [(t*gain,'Loading'),(-t*gain,'Unloading'),(t*0,'Holding')]:
            pred,details=classify(t,f,w,cfg);assert (pred[20:-20]==label).all()
            assert np.array_equal(pred,classify(t,f+100,w,cfg)[0])
            assert details['points']==round(w*30)+1
        step=np.where(t<5,0,gain);lab,_=classify(t,step,w,cfg)
        ind=np.flatnonzero(lab=='Loading');assert len(ind)>0 and np.max(abs(t[ind]-5))<=w+1/30
    raw=json.dumps(cfg,indent=2).encode();cp=OUT/'calibration.json'
    if cp.exists():assert cp.read_bytes()==raw,'Frozen rules cannot be revised in place'
    else:cp.write_bytes(raw)
    sha=hashlib.sha256(raw).hexdigest();parts=[];supports=[]
    for (seq,ep),g in d.groupby(['sequence','episode'],sort=False):
        g=g.sort_values('frame').copy()
        for w in WS:
            g[f'w{w:.1f}'],support=classify(g.timestamp.to_numpy(),g.signed_force.to_numpy(),w,cfg)
            supports.append({'sequence':seq,'episode':int(ep),**support})
        parts.append(g)
    d=pd.concat(parts,ignore_index=True);old=pd.read_parquet(PREV/'labels.parquet')
    d=d.merge(old[KEY+['original_phase','noise_1s','noise_2s']],on=KEY,validate='one_to_one',how='left',sort=False)
    assert d.original_phase.notna().all();d.to_parquet(OUT/'labels.parquet',index=False);pd.DataFrame(supports).to_csv(OUT/'window_support.csv',index=False)
    variants=[f'w{w:.1f}' for w in WS];stats=[]
    for seq,g in d.groupby('sequence'):
        for v in variants:
            changes=0;runs=[]
            for _,ep in g.groupby('episode'):
                a=ep[v].to_numpy();cuts=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,len(a)];changes+=len(cuts)-2
                runs.extend([(b-a)/30 for a,b in zip(cuts[:-1],cuts[1:])])
            stats.append({'sequence':seq,'window':v,'unknown_fraction':float((g[v]=='Transition / Unknown').mean()),'switches_per_minute':changes/(len(g)/30)*60,'median_run_seconds':float(np.median(runs)),'agreement_original':float((g[v]==g.original_phase).mean()),'agreement_1s':float((g[v]==g.noise_1s).mean()),**{ph:int((g[v]==ph).sum()) for ph in PHASES}})
    pd.DataFrame(stats).to_csv(OUT/'label_stability.csv',index=False)
    agree=[]
    for seq,g in d.groupby('sequence'):
        for v in variants:
            for v2 in variants:agree.append({'sequence':seq,'a':v,'b':v2,'agreement':float((g[v]==g[v2]).mean())})
    pd.DataFrame(agree).to_csv(OUT/'window_agreement.csv',index=False)
    records=[];probe=np.load(P/'N1_seed42_probe.npz');scale=probe['target_scale'];mean=probe['target_mean']
    for split in ['dev','test']:
        folder=P/f'{split}_bundle';meta=pd.read_parquet(folder/'metadata.parquet');y=np.load(folder/'targets.npy')
        labels=meta[KEY].merge(d[KEY+variants],on=KEY,how='left',sort=False,validate='one_to_one');assert labels[variants].notna().all().all()
        preds={}
        if split=='dev':
            for n in NS:
                for seed in SEEDS:
                    key=f'N{n}_seed{seed}';preds[key]=prediction(np.load(P/f'{key}_dev_embeddings.npy'),P/f'{key}_probe.npz')
        else:
            with np.load(P/'test_predictions.npz') as z:
                assert np.array_equal(z['targets'],y)
                for n in NS:
                    for seed in SEEDS:preds[f'N{n}_seed{seed}']=z[f'N{n}_seed{seed}']
        preds['train_mean']=np.broadcast_to(mean,y.shape)
        for key,pred in preds.items():
            assert np.isfinite(pred).all();per_frame=((pred[:,:6]-y[:,:6])/scale[:6])**2
            for seq in labels.sequence.unique():
                for v in variants:
                    for ph in ['ALL',*PHASES]:
                        mask=(labels.sequence==seq).to_numpy().copy()
                        if ph!='ALL':mask&=(labels[v]==ph).to_numpy()
                        if not mask.any():continue
                        records.append({'sequence':seq,'variant':v,'phase':ph,'model':key,'anchors':int(mask.sum()),'episodes':int(labels.loc[mask,'episode'].nunique()),'rmse':float(np.sqrt(per_frame[mask].mean()))})
    metrics=pd.DataFrame(records);metrics.to_csv(OUT/'metrics_per_seed.csv',index=False)
    a=metrics[metrics.model!='train_mean'].copy();a['N']=a.model.str.extract(r'N(\d+)').astype(int)
    agg=a.groupby(['sequence','variant','phase','N'],as_index=False).agg(rmse=('rmse','mean'),seed_sd=('rmse','std'),anchors=('anchors','first'),episodes=('episodes','first'));agg.to_csv(OUT/'comparison.csv',index=False)
    selection=[]
    for (v,ph),g in agg[agg.sequence=='gs_mini_obj2'].groupby(['variant','phase']):
        eligible=g.anchors.iloc[0]>=30 and g.episodes.iloc[0]>=2
        selection.append({'window':v,'phase':ph,'selected_N':int(g[g.rmse<=g.rmse.min()*1.05].N.min()) if eligible else None,'minimum_error_N':int(g.loc[g.rmse.idxmin(),'N']),'anchors':int(g.anchors.iloc[0]),'episodes':int(g.episodes.iloc[0])})
    pd.DataFrame(selection).to_csv(OUT/'development_selection.csv',index=False)
    # Same overall error for every new labelling, including previously inspected test.
    prev=json.loads((P/'results.json').read_text())
    for v in variants:
        for n in NS:
            row=agg[(agg.sequence=='gs_mini_obj4')&(agg.variant==v)&(agg.phase=='ALL')&(agg.N==n)].iloc[0]
            assert np.isclose(row.rmse,prev['test_aggregate'][str(n)]['ALL']['wrench_standardized_rmse_mean'],rtol=1e-6)
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,ph in zip(axs,PHASES[:3]):
        for v in variants:
            g=agg[(agg.sequence=='gs_mini_obj2')&(agg.variant==v)&(agg.phase==ph)]
            ax.plot(g.N,g.rmse,marker='o',label=v[1:]+'s')
        ax.set(title=ph,xlabel='Model input N',ylabel='Mean standardized wrench RMSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Development only: same models, different force-phase subsets');fig.tight_layout();fig.savefig(OUT/'development_N.png',dpi=150);plt.close(fig)
    review=pd.read_csv(PREV/'human_review.csv',keep_default_na=False);review.to_csv(OUT/'human_review.csv',index=False);(OUT/'review').mkdir(exist_ok=True)
    for idx,r in review.iterrows():
        g=d[(d.sequence==r.sequence)&(d.episode==r.episode)].reset_index(drop=True);mid=int(np.flatnonzero(g.frame.to_numpy()==r.frame)[0]);lo=max(0,mid-45);hi=min(len(g),mid+46);q=g.iloc[lo:hi]
        fig,axs=plt.subplots(2,1,figsize=(12,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        axs[0].plot(q.timestamp,q.signed_force,marker='.',ms=3,lw=.8);axs[0].axvline(g.timestamp.iloc[mid],color='black',ls='--');axs[0].set_ylabel('Signed, baseline-shifted RAW Fz');axs[0].set_title(f'{idx:02d} {r.sequence} episode {r.episode} frame {r.frame}; contact unverified')
        vs=['original_phase','noise_1s',*variants]
        for j,v in enumerate(vs):
            for ph,col in COLORS.items():
                m=q[v]==ph;axs[1].scatter(q.loc[m,'timestamp'],np.full(m.sum(),j),s=12,color=col)
        axs[1].set_yticks(range(len(vs)),['Original','1s']+[f'{w:.1f}s' for w in WS]);axs[1].set_xlabel('Episode timestamp (s)');fig.tight_layout();fig.savefig(OUT/'review'/f'{idx:02d}.png',dpi=130);plt.close(fig)
    hashes=json.loads((P/'freeze_hashes.json').read_text());assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in hashes.items());assert hashlib.sha256(cp.read_bytes()).hexdigest()==sha
    check={'synthetic_increase_decrease_constant_offset_step_checks':'PASS','overall_error_invariance':'PASS','original_probe_hashes':'PASS','calibration_sha256':sha,'human_ground_truth_accuracy':'not measured','models_retrained':False}
    (OUT/'verification.json').write_text(json.dumps(check,indent=2));print(json.dumps({'cfg':cfg,'selection':selection,'stats':stats},indent=2),flush=True)
    write_report(cfg,pd.DataFrame(stats),pd.DataFrame(selection),review)

def write_report(cfg,stats,selection,review):
    lines=[]
    for w in WS:
        name=f'w{w:.1f}';s=selection[selection.window==name];st=stats[(stats.sequence=='gs_mini_obj2')&(stats.window==name)].iloc[0]
        vals=[int(s[s.phase==ph].selected_N.iloc[0]) if pd.notna(s[s.phase==ph].selected_N.iloc[0]) else '不足' for ph in PHASES[:3]]
        lines.append(f'| {w:.1f}초 | {int(round(w*30))+1} | {st.unknown_fraction:.1%} | {st.switches_per_minute:.1f} | {vals[0]} | {vals[1]} | {vals[2]} |')
    report=f'''# 짧은 상황 판정 윈도우 비교

2026-11-24. 0.1·0.2·0.3·0.4·0.5초 다섯 설정을 같은 모델의 예측으로 비교했다. 모델 재학습과 기존 분류 수정은 없으며, 이 작업은 시간 구간 민감도 검증이다. 사람 정답 대비 분류 정확도는 아직 측정하지 않았다.

## 판정 방법과 시간 정렬

윈도우는 첫 표본과 마지막 표본 사이의 시간이다. 30fps에서는 0.1초에 3개 간격, 즉 4개 표본이 필요하다. 이전 설명의 약 3프레임은 프레임 빈도로 환산한 분량이며 끝점 간 시간 정의와 다르다. 이번에는 4·7·10·13·16개 표본으로 구분했다. 0.1·0.3·0.5초는 표본 수가 짝수라 앞보다 뒤를 한 프레임 더 포함하고, 창 중심이 평가 시점보다 약 0.0167초 뒤에 있다. 0.2·0.4초는 대칭이다. 실제 timestamp로 계산한 길이와 임계값은 window_support.csv에 있다.

원본 Fz를 사용해 기존 5프레임 및 추가 7프레임 평활화가 짧은 창보다 길어지는 문제를 피했다. 기존 episode별 부호·baseline은 유지했다. 각 창의 선형 회귀 기울기와 끝점 차이를 함께 사용한다. 상태 최소 지속 길이는 모든 창에서 2프레임(약 0.067초)으로 고정했다. 이전 0.4초 필터는 사용하지 않았다. 미래도 보는 오프라인 판정이다.

obj1·obj3 원신호의 2차 차분으로 추정한 노이즈 대용치는 {cfg['noise_sigma']:.5f}다. 변화량 문턱은 max(3√2×노이즈, 학습 힘 IQR×0.02), 기울기 문턱은 max(3×노이즈/√Σ시간편차², IQR×0.02/창길이)다. Holding은 두 문턱의 절반 이내이다. 학습 데이터로 기준을 고정한 후 예측 오차를 읽었다. 동작 자체가 노이즈 추정에 섞일 수 있고, 독립 잡음 가정도 근사이므로 물리적 노이즈 교정으로 해석하지 않는다.

## 개발 시퀀스 obj2 결과

| 창 길이 | 표본 수 | Unknown 비율 | 분당 라벨 전환 | Loading N | Holding N | Unloading N |
|---|---|---|---|---|---|---|
{chr(10).join(lines)}

N은 최소 30 anchor·2 episode가 있는 상태에서 최소 평균 오차의 5% 이내인 가장 작은 값이다. 정확히 최소 오차인 N은 development_selection.csv의 minimum_error_N 열로 별도 제공했다. 전체 데이터 N 선택은 라벨과 무관하다. Unknown 비율이나 전환 횟수가 작다고 정확한 분류라는 뜻은 아니다. 같은 상태명도 창마다 포함된 프레임이 달라서 오차가 낮아졌다고 분류 개선을 주장할 수 없다. 분당 전환 횟수는 Unknown으로 들어가고 나오는 전환도 센다.

comparison.csv에는 모든 물체별·상황별·N별 평균 및 seed 표준편차, metrics_per_seed.csv에는 평균값 기준선까지 있다. obj1·3은 학습 진단, obj2는 개발, obj4는 이미 열람한 테스트의 탐색적 재분석이다. 이 결과로 최종 분류 창을 선정하지 않았다. 강성 차이의 원인 분석이나 No Contact 검증도 아니다.

## 육안 확인과 검증 범위

index.html에서 이전과 동일한 20개 학습 구간의 짧은 창 라벨을 비교할 수 있다. 촉각 영상·압력 지도를 함께 보는 원래 검토 시트도 연결했다. 주황 Loading, 초록 Holding, 보라 Unloading, 회색 Unknown이다. CSV의 사람 판정 칸은 비어 있다.

합성 증가·감소·상수 신호, offset 불변성, 급격한 step 주변의 변화 감지, 표본 수, 동일 시점 매칭, 전체 오차 불변성, 원래 probe 해시 보존을 확인했다. 이것은 코드/민감도 검증이며 실제 상황 정답 검증은 아니다. 평활화와 지속 시간도 함께 바뀌었으므로 이전 1초 설정과의 차이는 창 길이만의 효과가 아니다. 이번 다섯 설정끼리는 동일한 전처리와 규칙을 사용했다.

다음에는 짧은 증가·감소와 노이즈성 흔들림을 구별할 수 있는 실제 구간을 확인해 분류 기준을 고정해야 한다. 모델 성능이 좋아 보이는 창을 정답 없이 고르면 상황 정의를 결과에 맞추는 문제가 생긴다.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'SHORT_WINDOW_RESULTS.md').write_text(report,encoding='utf-8')
    links=[]
    for i,r in review.iterrows():links.append(f'<li>{html.escape(str(r.sequence))} ep{r.episode} frame{r.frame}: <a href="review/{i:02d}.png">{i:02d} 짧은 창 비교</a> · <a href="../phase_audit_20261124/review/{i:02d}.png">촉각 이미지와 압력</a></li>')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>짧은 시간 창 비교</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:17px/1.7 sans-serif}li{margin:12px 0}img{width:100%}</style><h1>0.1–0.5초 상황 분류 비교</h1><p>주황: 증가 · 초록: 유지 · 보라: 감소 · 회색: 미확정. 접촉 여부 및 분류 정답은 미검증입니다. 08번 접촉 전후 후보부터 비교할 수 있습니다.</p><p><a href="RESULTS.md">결과 보고서</a> · <a href="label_stability.csv">분류 안정성</a> · <a href="development_selection.csv">N 선택표</a></p><ol>'+''.join(links)+'</ol><img src="development_N.png" alt="개발 데이터에서 창별 N 오차"></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8');print('SHORT WINDOW AUDIT COMPLETE',flush=True)

if __name__=='__main__':main()
