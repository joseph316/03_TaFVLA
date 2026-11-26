"""Train-only ridge probes, validation-only alpha/N selection, then quarantined holdout evaluation."""
import argparse,hashlib,json,gc,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.linalg import solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from script.stage2_train import FeatureDataset
from utils.checkpoint import load_checkpoint
from utils.model_factory import build_model

NS=[1,3,5,7,10];SEEDS=[42,43,44]
PHASES=['No Contact','Loading','Holding','Unloading']

def bundle(root,label_dir,out):
    if (out/'metadata.parquet').exists():
        return np.load(out/'features.npy',mmap_mode='r'),np.load(out/'targets.npy'),pd.read_parquet(out/'metadata.parquet')
    out.mkdir(parents=True,exist_ok=True)
    ds=FeatureDataset(Path(root).resolve(),window_size=10,stride=5,image_size=(224,224))
    x=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(len(ds),10,384))
    y=np.empty((len(ds),150),np.float32);records=[]
    for i in range(len(ds)):
        sample=ds[i];x[i]=sample['tactile'].numpy();y[i]=np.concatenate([sample['force'][-1].numpy(),sample['pressure'][-1].numpy().reshape(-1)])
        m=sample['metadata'];records.append({'sequence':m['sequence_id'],'episode':m['episode_index'],'frame':int(m['frame_indices'][-1]),'timestamp':float(m['timestamps'][-1])})
    x.flush();np.save(out/'targets.npy',y)
    labels=pd.read_parquet(Path(label_dir)/'labels.parquet')[['sequence','episode','frame','phase']]
    meta=pd.DataFrame(records).merge(labels,on=['sequence','episode','frame'],how='left',validate='one_to_one',sort=False)
    assert len(meta)==len(ds) and meta.phase.notna().all()
    meta.to_parquet(out/'metadata.parquet',index=False)
    return x,y,meta

def encode(path,x,n):
    ck=load_checkpoint(path)
    assert ck['step']==1000
    updates={int(v['step'].item()) for v in ck['optimizer']['state'].values() if 'step' in v}
    assert updates=={1000},updates
    assert ck['split']=={'train_groups':['gs_mini_obj1','gs_mini_obj3'],'validation_groups':['gs_mini_obj2']}
    model=build_model(ck['config'],pretrained=False).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    z=[]
    with torch.inference_mode():
        for start in range(0,len(x),256):
            raw=torch.from_numpy(np.array(x[start:start+256,-n:])).cuda()
            z.append(model.tactile_encoder.aggregate_features(model.tactile_encoder.projection(raw)).cpu().numpy())
    del model,ck;gc.collect();torch.cuda.empty_cache()
    return np.concatenate(z)

def metric(pred,target,scale,meta):
    rows={}
    for phase in ['ALL',*PHASES,'Transition / Unknown']:
        mask=np.ones(len(meta),bool) if phase=='ALL' else meta.phase.to_numpy()==phase
        count=int(mask.sum())
        if not count:rows[phase]={'count':0,'episodes':0};continue
        d=pred[mask]-target[mask]
        rows[phase]={'count':count,'episodes':int(meta.loc[mask,['sequence','episode']].drop_duplicates().shape[0]),
          'wrench_standardized_rmse':float(np.sqrt(np.mean((d[:,:6]/scale[:6])**2))),
          'wrench_raw_rmse_axes':np.sqrt(np.mean(d[:,:6]**2,axis=0)).tolist(),
          'pressure_raw_rmse':float(np.sqrt(np.mean(d[:,6:]**2))),
          'pressure_raw_mae':float(np.mean(abs(d[:,6:])))}
    return rows

def fit_probe(z,y,train,val,mean,scale,meta,path):
    xm=z[train].mean(0);xs=np.maximum(z[train].std(0),1e-6)
    a=((z[train]-xm)/xs).astype(np.float64);b=((y[train]-mean)/scale).astype(np.float64)
    # Explicit intercept after train-only centering.
    ym=b.mean(0);b=b-ym
    gram=a.T@a;cross=a.T@b;v=((z[val]-xm)/xs).astype(np.float64)
    trials=[];best=None
    for alpha in [.1,1.,10.,100.]:
        weights=solve(gram+alpha*np.eye(gram.shape[0]),cross,assume_a='pos')
        pred=((v@weights+ym)*scale+mean).astype(np.float32)
        score=float(np.sqrt(np.mean(((pred[:,:6]-y[val,:6])/scale[:6])**2)))
        trials.append({'alpha':alpha,'validation_wrench_standardized_rmse':score})
        if best is None or score<best[0]:best=(score,alpha,weights,pred)
    _,alpha,w,pred=best
    np.savez(path,x_mean=xm,x_std=xs,y_intercept=ym,weights=w,target_mean=mean,target_scale=scale,alpha=alpha)
    return {'alpha':alpha,'alpha_trials':trials,'metrics':metric(pred,y[val],scale,meta[val].reset_index(drop=True))},pred

def predict(z,path):
    p=np.load(path);return ((((z-p['x_mean'])/p['x_std'])@p['weights']+p['y_intercept'])*p['target_scale']+p['target_mean']).astype(np.float32)

def aggregate(records):
    result={}
    for n in NS:
        rows=[records[f'N{n}_seed{s}']['metrics'] for s in SEEDS];stats={}
        for phase in ['ALL',*PHASES,'Transition / Unknown']:
            base=rows[0][phase];stats[phase]={'count':base['count'],'episodes':base['episodes']}
            if base['count']:
                for k in ['wrench_standardized_rmse','pressure_raw_rmse']:
                    values=[r[phase][k] for r in rows];stats[phase][k+'_mean']=float(np.mean(values));stats[phase][k+'_sd']=float(np.std(values,ddof=1))
        result[str(n)]=stats
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('group',type=Path);args=parser.parse_args()
    torch.set_num_threads(4);group=args.group.resolve();out=group/'probes';out.mkdir(exist_ok=True)
    if (out/'frozen_selection.json').exists():raise RuntimeError('Selection already frozen; use existing artifacts, do not refit after test inspection')
    x,y,meta=bundle('../data/TaF-Dataset','../work/stage2_labels_dev',out/'dev_bundle')
    train=meta.sequence.isin(['gs_mini_obj1','gs_mini_obj3']).to_numpy();val=(meta.sequence=='gs_mini_obj2').to_numpy()
    assert np.all(train|val) and not np.any(train&val)
    mean=y[train].mean(0);scale=y[train].std(0);scale[:6]=np.maximum(scale[:6],1e-6);scale[6:]=max(float(np.sqrt(np.mean(scale[6:]**2))),1e-6)
    validation={};vpred={}
    # Baselines use exactly the same targets and sample endpoints.
    validation['train_mean']={'metrics':metric(np.broadcast_to(mean,y[val].shape),y[val],scale,meta[val].reset_index(drop=True))}
    for n in NS:
        key=f'ViT_mean{n}'
        z=np.array(x[:,-n:]).mean(1)
        validation[key],vpred[key]=fit_probe(z,y,train,val,mean,scale,meta,out/f'{key}_probe.npz')
    for n in NS:
        for seed in SEEDS:
            key=f'N{n}_seed{seed}';path=group/key/'last.pt'
            z=encode(path,x,n);np.save(out/f'{key}_dev_embeddings.npy',z)
            validation[key],vpred[key]=fit_probe(z,y,train,val,mean,scale,meta,out/f'{key}_probe.npz')
            print('PROBE VALIDATION',key,validation[key]['alpha'],validation[key]['metrics']['ALL'],flush=True)
    avg=aggregate(validation)
    def choose(phase):
        errors={n:avg[str(n)][phase]['wrench_standardized_rmse_mean'] for n in NS}
        floor=min(errors.values());return min(n for n in NS if errors[n]<=floor*1.05)
    global_n=choose('ALL');selected={};eligible={}
    for phase in PHASES:
        c=avg['1'][phase];eligible[phase]=c['count']>=30 and c['episodes']>=2
        selected[phase]=choose(phase) if eligible[phase] else None
    frozen={'global_N':global_n,'state_N':selected,'eligible_states':eligible,'rule':'smallest N within 5% of minimum mean validation standardized wrench RMSE','validation':validation,'validation_aggregate':avg,'train_anchors':int(train.sum()),'validation_anchors':int(val.sum()),'test_inspected':False,'label_config_sha256':hashlib.sha256(Path('../dataset_visualization-main/configs/classifier.yaml').read_bytes()).hexdigest()}
    # Freeze before reading any held-out labels, targets or results.
    frozen_path=out/'frozen_selection.json';frozen_path.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    probe_files=list(out.glob('*_probe.npz'));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in probe_files}
    hashes['frozen_selection.json']=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    (out/'freeze_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('SELECTION FROZEN',global_n,selected,flush=True)
    subprocess.run([sys.executable,'-u','-m','script.stage2_labels','--root','../data/TaF-Holdout','--out',str(out/'labels_test')],check=True)
    tx,ty,tm=bundle('../data/TaF-Holdout',out/'labels_test',out/'test_bundle')
    assert set(tm.sequence)=={'gs_mini_obj4'}
    test={'train_mean':{'metrics':metric(np.broadcast_to(mean,ty.shape),ty,scale,tm)}};tpred={}
    for n in NS:
        key=f'ViT_mean{n}';pred=predict(np.array(tx[:,-n:]).mean(1),out/f'{key}_probe.npz')
        tpred[key]=pred;test[key]={'metrics':metric(pred,ty,scale,tm)}
        for seed in SEEDS:
            key=f'N{n}_seed{seed}';z=encode(group/key/'last.pt',tx,n);pred=predict(z,out/f'{key}_probe.npz')
            tpred[key]=pred;test[key]={'metrics':metric(pred,ty,scale,tm)}
            print('TEST',key,test[key]['metrics']['ALL'],flush=True)
    routed={}
    for seed in SEEDS:
        pred=tpred[f'N{global_n}_seed{seed}'].copy()
        for phase,n in selected.items():
            if n is not None:
                mask=tm.phase.to_numpy()==phase;pred[mask]=tpred[f'N{n}_seed{seed}'][mask]
        routed[str(seed)]=metric(pred,ty,scale,tm)
    for name,sha in hashes.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
    # Persist predictions for independent auditing, not only aggregate scores.
    np.savez_compressed(out/'test_predictions.npz',targets=ty,target_scale=scale,**tpred)
    result={'frozen_global_N':global_n,'frozen_state_N':selected,'validation_aggregate':avg,'test':test,'test_aggregate':aggregate(test),'oracle_state_routed_test':routed,'test_anchors':len(tm),'test_episodes':int(tm.episode.nunique()),'checks':{'train_validation_disjoint':True,'identical_endpoints_all_N':True,'optimizer_updates_1000_all_runs':True,'probe_train_only':True,'selection_frozen_before_test':True,'freeze_hashes_unchanged':True},'limits':['Rule-based offline labels, not human ground truth','State routing uses held-out force-derived oracle labels; not an online controller','Single held-out sequence with few episodes','N changes both tactile and force alignment input context','No human or team agreement claimed']}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,source,title in [(axs[0],avg,'Development sequence: model selection'),(axs[1],result['test_aggregate'],'New holdout sequence: frozen evaluation')]:
        for phase in ['ALL','Loading','Holding','Unloading']:
            if source['1'][phase]['count']:
                vals=[source[str(n)][phase]['wrench_standardized_rmse_mean'] for n in NS];err=[source[str(n)][phase]['wrench_standardized_rmse_sd'] for n in NS]
                ax.errorbar(NS,vals,yerr=err,marker='o',capsize=3,label=phase)
        ax.set_title(title);ax.set_xlabel('N frames');ax.set_ylabel('Wrench standardized RMSE (lower is better)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'state_vs_N.png',dpi=160);plt.close(fig)
    print('PROBE PIPELINE COMPLETE; frozen selection retained',flush=True)
if __name__=='__main__':main()
