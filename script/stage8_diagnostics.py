"""Read-only error/distribution/convergence diagnostics; no post-test fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from script.stage3_phase_audit import prediction

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/stage2_20261120_115709';P=RUN/'probes';OUT=ROOT/'runs/error_diagnostics_20261124';AXES=['Fx','Fy','Fz','Tx','Ty','Tz']

def main():
    OUT.mkdir(exist_ok=True);m=pd.read_parquet(P/'dev_bundle/metadata.parquet');y=np.load(P/'dev_bundle/targets.npy');tm=pd.read_parquet(P/'test_bundle/metadata.parquet');ty=np.load(P/'test_bundle/targets.npy');train=m.sequence.isin(['gs_mini_obj1','gs_mini_obj3']).to_numpy();val=(m.sequence=='gs_mini_obj2').to_numpy()
    probe=np.load(P/'N3_seed42_probe.npz');mean=probe['target_mean'];scale=probe['target_scale'];tr=y[train];qlo,qhi=np.quantile(tr[:,:6],[.01,.99],axis=0);lo=tr[:,:6].min(0);hi=tr[:,:6].max(0)
    dist=[]
    for seq,values in [(s,y[(m.sequence==s).to_numpy()]) for s in m.sequence.unique()]+[('gs_mini_obj4',ty)]:
        for j,name in enumerate(AXES):
            a=values[:,j];dist.append({'sequence':seq,'axis':name,'count':len(a),'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'max':float(a.max()),'train_mean':float(mean[j]),'train_std':float(scale[j]),'mean_shift_train_sd':float((a.mean()-mean[j])/scale[j]),'outside_train_minmax_fraction':float(((a<lo[j])|(a>hi[j])).mean()),'outside_train_p01_p99_fraction':float(((a<qlo[j])|(a>qhi[j])).mean())})
    dist=pd.DataFrame(dist);dist.to_csv(OUT/'target_distribution.csv',index=False)
    errors=[];test_predictions=np.load(P/'test_predictions.npz')
    for split,target,meta in [('dev',y[val],m.loc[val].reset_index(drop=True)),('test',ty,tm)]:
        preds={'train_mean':np.broadcast_to(mean,target.shape)}
        for n in [1,3,5,7,10]:
            for seed in [42,43,44]:
                key=f'N{n}_seed{seed}';preds[key]=test_predictions[key] if split=='test' else prediction(np.load(P/f'{key}_dev_embeddings.npy')[val],P/f'{key}_probe.npz')
        for model,pred in preds.items():
            for ep in ['ALL',*sorted(meta.episode.unique())]:
                mask=np.ones(len(target),bool) if ep=='ALL' else (meta.episode==ep).to_numpy();a=target[mask,:6];b=pred[mask,:6];err=b-a;mse=np.mean(err**2,axis=0);stdmse=mse/scale[:6]**2
                for j,axis in enumerate(AXES):
                    v=float(np.var(a[:,j]));corr=float(np.corrcoef(a[:,j],b[:,j])[0,1]) if np.std(a[:,j])>1e-8 and np.std(b[:,j])>1e-8 and model!='train_mean' else None
                    errors.append({'split':split,'episode':str(ep),'model':model,'axis':axis,'raw_rmse':float(np.sqrt(mse[j])),'standardized_rmse':float(np.sqrt(stdmse[j])),'standardized_mse_share':float(stdmse[j]/stdmse.sum()),'bias_pred_minus_true':float(err[:,j].mean()),'bias_squared_fraction_mse':float(err[:,j].mean()**2/max(mse[j],1e-20)),'correlation':corr,'r2':float(1-mse[j]/v) if v>1e-20 else None})
    e=pd.DataFrame(errors);e.to_csv(OUT/'axis_errors.csv',index=False)
    # Verify aggregate standardized RMSE from axis MSE matches prior pipeline exactly.
    original=json.loads((P/'results.json').read_text())
    for n in [1,3,5,7,10]:
        for seed in [42,43,44]:
            key=f'N{n}_seed{seed}';g=e[(e.split=='test')&(e.episode=='ALL')&(e.model==key)];actual=np.sqrt(np.mean(g.standardized_rmse**2));expected=original['test'][key]['metrics']['ALL']['wrench_standardized_rmse'];assert np.isclose(actual,expected,rtol=1e-6)
    curves=[];summary=[]
    for n in [1,3,5,7,10]:
        for seed in [42,43,44]:
            rows=[json.loads(line) for line in (RUN/f'N{n}_seed{seed}/metrics.jsonl').read_text().splitlines()];training=[r for r in rows if 'train/loss_total' in r];validation=[r for r in rows if 'val/loss_total' in r]
            for r in training:curves.append({'N':n,'seed':seed,**r})
            first,last=training[0],training[-1];prior=training[-2]
            summary.append({'N':n,'seed':seed,'logged_training_points':len(training),'validation_points':len(validation),'final_step':last['step'],'final_epoch':last['epoch'],'nce_first':first['train/loss_nce'],'nce_final':last['train/loss_nce'],'nce_last_interval_change':last['train/loss_nce']-prior['train/loss_nce'],'reconstruction_first':first['train/loss_recon'],'reconstruction_final':last['train/loss_recon'],'reconstruction_last_interval_change':last['train/loss_recon']-prior['train/loss_recon'],'val_evaluated_samples':validation[-1].get('val/evaluated_samples') if validation else None,'final_force_batch_codebook_usage':last.get('train/force_codebook_usage'),'final_pressure_batch_codebook_usage':last.get('train/pressure_codebook_usage')})
    pd.DataFrame(curves).to_csv(OUT/'training_curves.csv',index=False);s=pd.DataFrame(summary);s.to_csv(OUT/'training_summary.csv',index=False)
    core=e[(e.episode=='ALL')&e.model.str.startswith('N3_seed')].groupby(['split','axis'],as_index=False).agg(standardized_rmse=('standardized_rmse','mean'),raw_rmse=('raw_rmse','mean'),mse_share=('standardized_mse_share','mean'),bias_fraction=('bias_squared_fraction_mse','mean'),correlation=('correlation','mean'));core.to_csv(OUT/'N3_axis_summary.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,split in zip(axs,['dev','test']):
        g=core[core.split==split].set_index('axis').loc[AXES];base=e[(e.split==split)&(e.model=='train_mean')&(e.episode=='ALL')].set_index('axis').loc[AXES];x=np.arange(6);ax.bar(x-.18,g.standardized_rmse,width=.36,label='N3 seed-mean axis RMSE');ax.bar(x+.18,base.standardized_rmse,width=.36,label='Training mean');ax.set_xticks(x,AXES);ax.set_title(split);ax.set_ylabel('Axis standardized RMSE');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'axis_errors.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));c=pd.DataFrame(curves)
    for seed in [42,43,44]:
        g=c[(c.N==3)&(c.seed==seed)]
        for ax,col in zip(axs,['train/loss_nce','train/loss_recon']):ax.plot(g.step,g[col],marker='o',label=str(seed));ax.set_xlabel('Optimizer updates');ax.set_ylabel(col);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'training_N3.png',dpi=150);plt.close(fig)
    test_predictions.close();hashes=json.loads((P/'freeze_hashes.json').read_text());assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    (OUT/'verification.json').write_text(json.dumps({'aggregate_axis_reconstruction':'PASS for all 15 test models','original_probe_hashes':'PASS','refitting':False},indent=2))
    print('TEST N3 AXIS SUMMARY\n'+core[core.split=='test'].to_string(index=False));print('TEST TARGET DISTRIBUTION\n'+dist[dist.sequence=='gs_mini_obj4'].to_string(index=False));print('TRAINING SUMMARY\n'+s.to_string(index=False))

if __name__=='__main__':main()
