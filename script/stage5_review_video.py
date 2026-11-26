"""Synchronized review clips from training sequences only; no human truth fabricated."""
from pathlib import Path
import json,html,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from script.local_rgb_cache import CachedTaFDataset
from script.stage3_phase_audit import COLORS

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/video_review_20261124'
SHORT=ROOT/'runs/short_windows_20261124'

def main():
    OUT.mkdir(exist_ok=True);plt.rcParams['animation.ffmpeg_path']=str(ROOT/'tools/ffmpeg/bin/ffmpeg.exe')
    labels=pd.read_parquet(SHORT/'labels.parquet');review=pd.read_csv(SHORT/'human_review.csv',keep_default_na=False)
    tasks=[]
    for i,title in [(8,'contact_transition'),(0,'steady_force'),(2,'increasing_force'),(6,'decreasing_force')]:
        r=review.iloc[i];tasks.append({'id':f'{i:02d}_{title}','sequence':r.sequence,'episode':int(r.episode),'frame':int(r.frame),'reason':title})
    best=None
    for (seq,ep),g in labels[labels.sequence.isin(['gs_mini_obj1','gs_mini_obj3'])].groupby(['sequence','episode']):
        g=g.sort_values('frame').reset_index(drop=True);diff=(g['w0.2']!=g['w0.3']).astype(float)
        score=diff.rolling(90,center=True,min_periods=90).mean();idx=score.idxmax()
        if pd.notna(idx) and (best is None or score.loc[idx]>best[0]):best=(score.loc[idx],seq,int(ep),int(g.loc[idx,'frame']))
    tasks.append({'id':'disagreement','sequence':best[1],'episode':best[2],'frame':best[3],'reason':'maximum 3s disagreement between 0.2s and 0.3s on training data','disagreement_fraction':float(best[0])})
    ds=CachedTaFDataset(ROOT/'data/TaF-Dataset',window_size=10,stride=5,image_size=(224,224));eps={(e.sequence_id,e.episode_index):e for e in ds.episodes}
    manifest=[]
    for task in tasks:
        ep=eps[(task['sequence'],task['episode'])];g=labels[(labels.sequence==task['sequence'])&(labels.episode==task['episode'])].sort_values('frame').reset_index(drop=True)
        assert np.array_equal(g.frame,ep.frames)
        mid=int(np.flatnonzero(ep.frames==task['frame'])[0]);start=max(0,mid-90);stop=min(ep.length,mid+91);offsets=np.arange(start,stop)
        images,_=ds._images(ep,offsets);_,pressure=ds._sensor_rows(ep,offsets);q=g.iloc[offsets].reset_index(drop=True)
        time=q.timestamp.to_numpy();dt=np.diff(time);assert np.allclose(dt,1/30,atol=2e-5)
        fig=plt.figure(figsize=(12,8),dpi=100);grid=fig.add_gridspec(3,2,height_ratios=[2.2,1.3,1.2])
        axim=fig.add_subplot(grid[0,0]);im=axim.imshow(images[0]);axim.axis('off');axim.set_title('Tactile image')
        axp=fig.add_subplot(grid[0,1]);pm=axp.imshow(pressure[0],cmap='magma',vmin=float(pressure.min()),vmax=float(pressure.max()));fig.colorbar(pm,ax=axp,fraction=.045);axp.set_title('Raw pressure: fixed scale throughout clip')
        axf=fig.add_subplot(grid[1,:]);axf.plot(time,q.signed_force,lw=1);axf.set_ylabel('Signed RAW Fz');axf.set_xlim(time[0],time[-1]);cursor=axf.axvline(time[0],color='red',lw=2);dot,=axf.plot([time[0]],[q.signed_force.iloc[0]],'ro');axf.grid(alpha=.2)
        axl=fig.add_subplot(grid[2,:],sharex=axf);variants=['original_phase','noise_1s','w0.1','w0.2','w0.3','w0.4','w0.5']
        for j,v in enumerate(variants):
            for ph,col in COLORS.items():
                mask=q[v]==ph;axl.scatter(time[mask],np.full(mask.sum(),j),c=col,s=9,label=ph if j==0 else None)
        axl.set_yticks(range(7),['Original','1s','0.1s','0.2s','0.3s','0.4s','0.5s']);axl.set_xlabel('Episode timestamp (seconds)');cursor2=axl.axvline(time[0],color='red',lw=2);axl.legend(ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.25))
        heading=fig.suptitle('');fig.subplots_adjust(top=.91,bottom=.12,hspace=.38)
        video=OUT/(task['id']+'.mp4');writer=FFMpegWriter(fps=30,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','21','-movflags','+faststart'])
        with writer.saving(fig,str(video),100):
            for k in range(len(q)):
                im.set_data(images[k]);pm.set_data(pressure[k]);cursor.set_xdata([time[k],time[k]]);cursor2.set_xdata([time[k],time[k]]);dot.set_data([time[k]],[q.signed_force.iloc[k]])
                heading.set_text(f"{task['id']} | {task['sequence']} ep{task['episode']} | frame {q.frame.iloc[k]} | t={time[k]:.3f}s\nOrange: increasing; green: steady; purple: decreasing; grey: uncertain. Contact unverified.")
                writer.grab_frame()
                if k==mid-start:fig.savefig(OUT/(task['id']+'.png'),dpi=100)
        plt.close(fig)
        probe=json.loads(subprocess.check_output([str(ROOT/'tools/ffmpeg/bin/ffprobe.exe'),'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,avg_frame_rate,width,height:format=duration','-of','json',str(video)],text=True))
        assert int(probe['streams'][0]['nb_read_frames'])==len(q);assert probe['streams'][0]['avg_frame_rate']=='30/1'
        assert abs(float(probe['format']['duration'])-len(q)/30)<.05
        q.to_csv(OUT/(task['id']+'_frames.csv'),index=False)
        manifest.append({**task,'start_frame':int(q.frame.iloc[0]),'end_frame':int(q.frame.iloc[-1]),'start_timestamp':float(time[0]),'end_timestamp':float(time[-1]),'frames':len(q),'video_duration_seconds':float(probe['format']['duration']),'fps':30,'source_video':str(ep.video),'verified_encoded_frame_count':True})
        print('VIDEO COMPLETE',task['id'],len(q),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    instructions={'contact_transition':'급증 전의 평평한 구간을 실제 비접촉으로 볼 수 있는지, 급증 직후 하강이 힘을 빼는 동작인지 일시적인 충격인지 확인합니다.','steady_force':'촉각·압력 분포가 유지되는 동안 짧은 창이 불필요하게 상태를 바꾸는지 확인합니다.','increasing_force':'지속적인 힘 증가의 시작·끝이 영상·압력 변화와 맞는지 확인합니다.','decreasing_force':'힘 감소와 압력 변화가 함께 나타나는지, 경계가 지나치게 앞당겨지는지 확인합니다.'}
    cards=[]
    for t in tasks:
        desc=instructions.get(t['reason'],'0.2초와 0.3초의 라벨이 가장 많이 다른 학습 구간입니다. 실제 변화를 놓치는지 또는 작은 흔들림에 과민한지 확인합니다.')
        cards.append(f'<section><h2>{html.escape(t["id"])}</h2><p>{desc}</p><video controls preload="metadata" poster="{t["id"]}.png" src="{t["id"]}.mp4"></video><p><a href="{t["id"]}_frames.csv">프레임별 수치와 라벨</a></p></section>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>상황 분류 동영상 검토</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:17px/1.7 sans-serif;background:#f5f6f8;color:#18202b}section{background:white;padding:20px;margin:25px 0;border-radius:10px}video{width:100%}button{padding:8px 16px;margin:5px}</style><h1>상황 분류 동영상 검토</h1><p>학습 데이터의 대표 구간 5개입니다. 빨간 선은 영상과 동기화된 현재 시점입니다. 아래 라벨은 미래 정보도 참고하는 오프라인 판정이며 정답이 아닙니다. 초록은 힘이 일정하다는 뜻이며 접촉 여부를 보증하지 않습니다.</p><p>주황: 증가 · 초록: 유지 · 보라: 감소 · 회색: 미확정.</p><p>전체 재생 속도: <button onclick="speed(.25)">0.25배</button><button onclick="speed(.5)">0.5배</button><button onclick="speed(1)">1배</button></p><p><a href="REVIEW_NOTES.md">검토 메모</a> · <a href="review_decisions.csv">판정 기록 CSV</a></p>'''+''.join(cards)+'''<script>function speed(n){document.querySelectorAll('video').forEach(v=>v.playbackRate=n)}</script></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    pd.DataFrame([{'clip':t['id'],'human_phase_interpretation':'','human_contact':'','impact_or_release':'','preferred_window':'','notes':''} for t in tasks]).to_csv(OUT/'review_decisions.csv',index=False)
    print('REVIEW PACKAGE COMPLETE',flush=True)

if __name__=='__main__':main()
