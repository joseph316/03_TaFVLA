"""Exploratory offline relabelling; never modifies upstream labels or refits models.

No verified no-contact reference exists: phase and contact status stay separate.
Fit noise/scale on obj1+obj3 only. Preserve episode sign convention from stage2.
"""
from pathlib import Path
import json, hashlib, os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'runs/stage2_20261120_115709/probes'
OUT=ROOT/'runs/phase_audit_20261124'
NS=[1,3,5,7,10]; SEEDS=[42,43,44]
KEY=['sequence','episode','frame']
PHASES=['Loading','Holding','Unloading','Transition / Unknown']
COLORS={'Loading':'#e66101','Holding':'#1b9e77','Unloading':'#7570b3','Transition / Unknown':'#b8b8b8'}

def robust_sigma(a):
    return float(1.4826*np.median(np.abs(a-np.median(a))))

def classify(t,f,window,cfg):
    t=np.asarray(t);f=np.asarray(f);dt=float(np.median(np.diff(t)))
    w=max(5,int(round(window/dt))|1);h=w//2
    smooth=savgol_filter(f,min(7,len(f)//2*2-1),2)
    x=np.arange(-h,h+1)*dt
    slope=np.convolve(smooth,x[::-1]/np.sum(x*x),'same')
    change=np.full(len(f),np.nan);change[h:-h]=smooth[2*h:]-smooth[:-2*h]
    duration=2*h*dt
    threshold=max(cfg['change_noise_multiplier']*cfg['noise_sigma'],cfg['range_fraction']*cfg['force_iqr'])
    loading=(change>=threshold)&(slope>=threshold/duration)
    unloading=(change<=-threshold)&(slope<=-threshold/duration)
    holding=(abs(change)<=threshold*.5)&(abs(slope)<=threshold*.5/duration)
    labels=np.full(len(f),'Transition / Unknown',dtype=object)
    labels[loading]='Loading';labels[unloading]='Unloading';labels[holding]='Holding'
    labels[:h]='Transition / Unknown';labels[-h:]='Transition / Unknown'
    cuts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if (b-a)*dt<cfg['min_duration_seconds']:labels[a:b]='Transition / Unknown'
    return labels,slope,change,threshold

def tests(cfg):
    t=np.arange(301)/30;gain=max(10.,10*cfg['force_iqr'],100*cfg['noise_sigma'])
    for window in [1.,2.]:
        for signal,name in [(t*gain,'Loading'),(-t*gain,'Unloading'),(t*0+100,'Holding')]:
            labels,*_=classify(t,signal,window,cfg)
            assert np.all(labels[70:-70]==name),(window,name)
            shifted,*_=classify(t,signal+123,window,cfg)
            assert np.array_equal(labels,shifted),'Baseline shift changed phase'
        assert not np.any(classify(t,t*0,window,cfg)[0]=='No Contact')

def label_table(df,cfg):
    parts=[]
    for _,g in df.groupby(['sequence','episode'],sort=False):
        g=g.sort_values('frame').copy();g['original_phase']=g['phase'];g['contact_status']='Unverified'
        for window in [1.,2.]:
            labels,slope,change,thr=classify(g.timestamp.to_numpy(),g.normal_force.to_numpy(),window,cfg)
            name=f'noise_{int(window)}s';g[name]=labels;g[name+'_slope']=slope;g[name+'_change']=change
        parts.append(g)
    return pd.concat(parts,ignore_index=True)

def prediction(z,path):
    a=np.load(path)
    return ((((z-a['x_mean'])/a['x_std'])@a['weights']+a['y_intercept'])*a['target_scale']+a['target_mean']).astype(np.float32)

def main():
    OUT.mkdir(exist_ok=True)
    dev=pd.read_parquet(ROOT/'work/stage2_labels_dev/labels.parquet')
    train=dev[dev.sequence.isin(['gs_mini_obj1','gs_mini_obj3'])]
    noises=[]
    for _,g in train.groupby(['sequence','episode']):
        noises.append(robust_sigma(np.diff(g.normal_force.to_numpy(),n=2))/np.sqrt(6))
    cfg={'fit_sequences':['gs_mini_obj1','gs_mini_obj3'],'noise_sigma':float(np.median(noises)),
         'noise_estimator':'median episode MAD(second difference of existing lightly smoothed force)/sqrt(6); proxy, not calibrated sensor noise',
         'force_iqr':float(train.normal_force.quantile(.75)-train.normal_force.quantile(.25)),
         'change_noise_multiplier':6.,'range_fraction':.02,'min_duration_seconds':.4,
         'windows_seconds':[1.,2.], 'primary_variant':'noise_1s','secondary_variant':'noise_2s',
         'contact_status':'Unverified; no No Contact assigned without verified reference',
         'limitations':['Offline centered windows','Inherited episode-wide auto sign and baseline','No object-specific normalization','Heuristic thresholds, no human accuracy measurement','Previously inspected obj4: exploratory reanalysis only']}
    tests(cfg)
    raw=json.dumps(cfg,indent=2).encode()
    if (OUT/'calibration.json').exists():
        assert (OUT/'calibration.json').read_bytes()==raw,'Cannot change frozen calibration; use new audit directory'
    else:(OUT/'calibration.json').write_bytes(raw)
    frozen=hashlib.sha256(raw).hexdigest()
    labelled=label_table(dev,cfg)
    # No target-model errors are read until rules and calibration have been saved.
    held=pd.read_parquet(P/'labels_test/labels.parquet');labelled=pd.concat([labelled,label_table(held,cfg)],ignore_index=True)
    labelled.to_parquet(OUT/'labels.parquet',index=False)
    variants=['original_phase','noise_1s','noise_2s'];counts=[];agreement=[]
    for seq,g in labelled.groupby('sequence'):
        for v in variants:
            for phase,count in g[v].value_counts().items():counts.append({'sequence':seq,'variant':v,'phase':phase,'frames':int(count),'fraction':count/len(g)})
        agreement.append({'sequence':seq,'original_vs_1s':float((g.original_phase==g.noise_1s).mean()),'original_vs_2s':float((g.original_phase==g.noise_2s).mean()),'1s_vs_2s':float((g.noise_1s==g.noise_2s).mean())})
        pd.crosstab(g.original_phase,g.noise_1s).to_csv(OUT/f'{seq}_confusion.csv')
    pd.DataFrame(counts).to_csv(OUT/'phase_counts.csv',index=False);pd.DataFrame(agreement).to_csv(OUT/'agreement.csv',index=False)
    # Purely descriptive low-pressure reference candidates; not contact ground truth.
    candidates=[]
    for (seq,ep),g in labelled[labelled.sequence.isin(cfg['fit_sequences'])].groupby(['sequence','episode']):
        for v in ['noise_1s']:
            cuts=np.r_[0,np.flatnonzero(g[v].to_numpy()[1:]!=g[v].to_numpy()[:-1])+1,len(g)]
            for a,b in zip(cuts[:-1],cuts[1:]):
                if b-a>=30:
                    middle=g.iloc[(a+b)//2]
                    candidates.append({'sequence':seq,'episode':int(ep),'frame':int(middle.frame),'start_frame':int(g.iloc[a].frame),'end_frame':int(g.iloc[b-1].frame),'suggested_phase':g.iloc[a][v],'kind':'phase','duration':(b-a)/30,'human_phase':'','human_contact':'','notes':''})
        low=g.iloc[int(np.argmin(g.pressure_total.to_numpy()))]
        candidates.append({'sequence':seq,'episode':int(ep),'frame':int(low.frame),'start_frame':int(low.frame),'end_frame':int(low.frame),'suggested_phase':'Unverified','kind':'low_pressure_candidate','duration':0,'human_phase':'','human_contact':'','notes':'Minimum pressure is not proof of no contact'})
    c=pd.DataFrame(candidates);chosen=[]
    for (_,phase),g in c.groupby(['sequence','suggested_phase']):chosen.append(g.sort_values('duration',ascending=False).drop_duplicates(['episode']).head(2))
    review=pd.concat(chosen,ignore_index=True);review.to_csv(OUT/'human_review.csv',index=False)
    results=[];example=np.load(P/'N1_seed42_probe.npz');scale=example['target_scale'];mean=example['target_mean']
    for split in ['dev','test']:
        folder=P/f'{split}_bundle';meta=pd.read_parquet(folder/'metadata.parquet')
        joined=meta[KEY].merge(labelled[KEY+variants],on=KEY,how='left',sort=False,validate='one_to_one')
        assert len(joined)==len(meta) and joined[variants].notna().all().all()
        y=np.load(folder/'targets.npy');preds={}
        if split=='dev':
            for n in NS:
                for s in SEEDS:
                    key=f'N{n}_seed{s}';preds[key]=prediction(np.load(P/f'{key}_dev_embeddings.npy'),P/f'{key}_probe.npz')
        else:
            with np.load(P/'test_predictions.npz') as z:
                assert np.array_equal(z['targets'],y)
                for n in NS:
                    for s in SEEDS:preds[f'N{n}_seed{s}']=z[f'N{n}_seed{s}']
        preds['train_mean']=np.broadcast_to(mean,y.shape)
        for seq in joined.sequence.unique():
            for v in variants:
                for phase in ['ALL',*PHASES]:
                    mask=(joined.sequence==seq).to_numpy().copy()
                    if phase!='ALL':mask&=(joined[v]==phase).to_numpy()
                    if not mask.any():continue
                    for key,pred in preds.items():
                        assert np.isfinite(pred).all()
                        err=(pred[mask,:6]-y[mask,:6])/scale[:6]
                        results.append({'sequence':seq,'split_role':'train_diagnostic' if seq in cfg['fit_sequences'] else ('development' if split=='dev' else 'previously_seen_test'),'variant':v,'phase':phase,'model':key,'anchors':int(mask.sum()),'episodes':int(joined.loc[mask,'episode'].nunique()),'wrench_rmse':float(np.sqrt(np.mean(err**2)))})
    result=pd.DataFrame(results);result.to_csv(OUT/'metrics_per_seed.csv',index=False)
    adapters=result[result.model!='train_mean'].copy();adapters['N']=adapters.model.str.extract(r'N(\d+)').astype(int)
    agg=adapters.groupby(['sequence','split_role','variant','phase','N'],as_index=False).agg(rmse=('wrench_rmse','mean'),seed_sd=('wrench_rmse','std'),anchors=('anchors','first'),episodes=('episodes','first'))
    agg.to_csv(OUT/'comparison.csv',index=False)
    selected=[]
    for (v,phase),g in agg[agg.sequence=='gs_mini_obj2'].groupby(['variant','phase']):
        eligible=int(g.anchors.iloc[0])>=30 and int(g.episodes.iloc[0])>=2
        n=int(g[g.rmse<=g.rmse.min()*1.05].N.min()) if eligible else None
        selected.append({'variant':v,'phase':phase,'N_within_5_percent':n,'anchors':int(g.anchors.iloc[0]),'episodes':int(g.episodes.iloc[0])})
    pd.DataFrame(selected).to_csv(OUT/'development_selection.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,seq in zip(axs.flat,sorted(labelled.sequence.unique())):
        for phase in ['Loading','Holding','Unloading']:
            g=agg[(agg.sequence==seq)&(agg.variant=='noise_1s')&(agg.phase==phase)]
            ax.errorbar(g.N,g.rmse,yerr=g.seed_sd,marker='o',capsize=3,color=COLORS[phase],label=f'{phase} (n={int(g.anchors.iloc[0]) if len(g) else 0})')
        ax.set_title(seq+(' [train diagnostic]' if seq in cfg['fit_sequences'] else ''));ax.set_xlabel('N');ax.set_ylabel('Wrench standardized RMSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Noise-aware 1s force phases; contact unverified; bars = seed SD');fig.tight_layout();fig.savefig(OUT/'object_phase_N.png',dpi=150);plt.close(fig)
    assert hashlib.sha256((OUT/'calibration.json').read_bytes()).hexdigest()==frozen
    hashes=json.loads((P/'freeze_hashes.json').read_text());assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    (OUT/'verification.json').write_text(json.dumps({'synthetic_trend_and_offset_tests':'PASS','original_probes_unchanged':True,'calibration_sha256':frozen,'no_training_performed':True,'endpoints_matched':True,'human_validation':'pending'},indent=2))
    print(json.dumps({'calibration':cfg,'agreement':agreement,'dev_selection':selected},indent=2),flush=True)
    # Produce review sheets from the original frame-index mapping and cached images.
    from script.local_rgb_cache import CachedTaFDataset
    ds=CachedTaFDataset(ROOT/'data/TaF-Dataset',window_size=10,stride=5,image_size=(224,224))
    eps={(e.sequence_id,e.episode_index):e for e in ds.episodes}
    sheets=OUT/'review';sheets.mkdir(exist_ok=True)
    for idx,row in review.iterrows():
        ep=eps[(row.sequence,row.episode)];g=labelled[(labelled.sequence==row.sequence)&(labelled.episode==row.episode)].reset_index(drop=True)
        mid=int(np.flatnonzero(ep.frames==row.frame)[0]);offsets=np.clip(np.array([mid-15,mid,mid+15]),0,ep.length-1)
        imgs,_=ds._images(ep,offsets);_,pressure=ds._sensor_rows(ep,offsets)
        fig=plt.figure(figsize=(11,8));grid=fig.add_gridspec(4,3,height_ratios=[1,1,1,.4])
        for j in range(3):
            ax=fig.add_subplot(grid[0,j]);ax.imshow(imgs[j]);ax.set_title(f'frame {ep.frames[offsets[j]]}');ax.axis('off')
            ax=fig.add_subplot(grid[1,j]);im=ax.imshow(pressure[j],vmin=float(pressure.min()),vmax=float(pressure.max()),cmap='magma');fig.colorbar(im,ax=ax,fraction=.046);ax.set_title('Raw pressure')
        lo=max(0,mid-90);hi=min(len(g),mid+91);t=g.timestamp.to_numpy()
        ax=fig.add_subplot(grid[2,:]);ax.plot(t[lo:hi],g.normal_force.iloc[lo:hi]);ax.axvline(t[mid],color='black',ls='--');ax.set_ylabel('Preprocessed Fz')
        ax=fig.add_subplot(grid[3,:])
        for y,v in enumerate(['original_phase','noise_1s','noise_2s']):
            for ph,color in COLORS.items():
                mask=g[v].iloc[lo:hi].to_numpy()==ph;ax.scatter(t[lo:hi][mask],np.full(mask.sum(),y),s=12,color=color)
        ax.set_yticks([0,1,2],['Original','1s','2s']);ax.set_xlabel('Episode timestamp (s)')
        fig.suptitle(f'{idx:02d} {row.sequence} ep{row.episode}: {row.suggested_phase}\nHuman phase/contact labels pending; image alone may not resolve contact')
        fig.tight_layout();fig.savefig(sheets/f'{idx:02d}.png',dpi=115);plt.close(fig)
    print('AUDIT AND REVIEW SHEETS COMPLETE',flush=True)

if __name__=='__main__':main()
