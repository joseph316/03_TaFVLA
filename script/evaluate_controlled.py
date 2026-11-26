"""Evaluate controlled pilots on identical endpoints; keep learned-code metrics diagnostic."""
import argparse,json,time,gc
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Subset
from script.train_controlled import MatchedDataset
from utils.checkpoint import load_checkpoint
from utils.model_factory import build_model
from utils.alignment import compute_alignment_metrics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('group',type=Path);args=parser.parse_args()
    group=args.group.resolve();out=group/'comparison';out.mkdir(exist_ok=True)
    dataset=MatchedDataset(Path('../data/TaF-Dataset').resolve(),window_size=5,stride=5,image_size=(224,224))
    checkpoints={name:group/name/'last.pt' for name in ['N5_cell','N5_pooled','N1_pooled']}
    shared_indices=None;shared_split=None;norms={};results={};endpoint_reference=None
    for name,path in checkpoints.items():
        ck=load_checkpoint(path);cfg=ck['config'];n=cfg['data']['window_size']
        assert ck['step']==1000
        updates=sorted({int(v['step'].item()) for v in ck['optimizer']['state'].values() if 'step' in v})
        assert updates==[1000],updates
        assert all(torch.isfinite(t).all() for t in ck['model'].values() if t.is_floating_point())
        indices=ck['alignment']['sample_indices']
        if shared_indices is None:shared_indices=indices;shared_split=ck['split']
        assert indices==shared_indices and ck['split']==shared_split
        norms[name]={k:v.numpy() for k,v in ck['normalization'].items()}
        dataset.requested_window=n
        model=build_model(cfg,pretrained=False).cuda().eval();model.load_state_dict(ck['model'],strict=True)
        zs=[];cs=[];codes=[];pressure_sq=[];force_sq=[];pressure_mae=[];endpoints=[];base_ps=[];base_fs=[];raw_pressures=[];raw_forces=[]
        loader=DataLoader(Subset(dataset,indices),batch_size=16,shuffle=False,num_workers=0)
        with torch.inference_mode():
            for batch in loader:
                x={k:batch[k].cuda() for k in ['tactile','force','pressure']}
                raw_pressures.append(x['pressure'][:,-1].cpu().flatten(1));raw_forces.append(x['force'][:,-1].cpu())
                pred=model(**x)
                assert x['tactile'].shape[1]==n
                zs.append(pred['z_tactile'].cpu());cs.append(pred['c_force'].cpu())
                codes.append(torch.stack([pred['codebook_indices']['pressure'],pred['codebook_indices']['force']],dim=1).cpu())
                dp=(pred['reconstructed_pressure'][:,-1]-x['pressure'][:,-1]).cpu()
                df=(pred['reconstructed_force'][:,-1]-x['force'][:,-1]).cpu()
                pressure_sq.append(dp.square().flatten());pressure_mae.append(dp.abs().flatten());force_sq.append(df.square())
                base_ps.append((x['pressure'][:,-1].cpu()-ck['normalization']['pressure_mean']).square().flatten())
                base_fs.append((x['force'][:,-1].cpu()-ck['normalization']['force_mean']).square())
                endpoints.extend(zip(batch['metadata']['sequence_id'],batch['metadata']['episode_index'].tolist(),batch['metadata']['frame_indices'][:,-1].tolist()))
            if endpoint_reference is None:endpoint_reference=endpoints
            assert endpoints==endpoint_reference
            z=torch.cat(zs);c=torch.cat(cs);q=torch.cat(codes)
            alignment=compute_alignment_metrics(z,c);sim=alignment.similarity
            unique,counts=torch.unique(q,dim=0,return_counts=True)
            choices=sim.argmax(dim=1)
            p_raw=torch.cat(raw_pressures);f_raw=torch.cat(raw_forces)
            p_distance=torch.cdist(p_raw,p_raw).square()/144
            f_scaled=f_raw/ck['normalization']['force_std']
            f_distance=torch.cdist(f_scaled,f_scaled).square()/6
            eye=torch.eye(len(q),dtype=torch.bool);rows_id=torch.arange(len(q))
            retrieved_pressure_rmse=p_distance[rows_id,choices].mean().sqrt().item()
            retrieved_force_std_rmse=f_distance[rows_id,choices].mean().sqrt().item()
            random_pressure_rmse=p_distance[~eye].mean().sqrt().item()
            random_force_std_rmse=f_distance[~eye].mean().sqrt().item()
            code_match=(q[choices]==q).all(dim=1).float().mean().item()
            chance=(counts.float()/len(q)).square().sum().item()
            # Isolate tactile-only model compute: warmed batch=16, no loading or transfer.
            tactile=x['tactile']
            for _ in range(10):model.encode_tactile(tactile)
            torch.cuda.synchronize();timings=[]
            for _ in range(50):
                start=time.perf_counter();model.encode_tactile(tactile);torch.cuda.synchronize()
                timings.append((time.perf_counter()-start)*1000)
        rows=[json.loads(line) for line in (path.parent/'metrics.jsonl').read_text().splitlines()]
        train=[r for r in rows if 'train/loss_total' in r];val=[r for r in rows if 'val/loss_total' in r]
        assert all(np.isfinite(r['grad_norm']) for r in train)
        results[name]={'model_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'window':n,'optimizer_updates':updates[0],'split':ck['split'],'pair_count':len(q),
          'retrieved_candidate_pressure_rmse_raw':retrieved_pressure_rmse,'random_other_candidate_pressure_rmse_raw':random_pressure_rmse,
          'retrieved_candidate_force_standardized_rmse':retrieved_force_std_rmse,'random_other_candidate_force_standardized_rmse':random_force_std_rmse,
          'exact_retrieval':alignment.metrics,'unique_pressure_codes':len(torch.unique(q[:,0])),
          'unique_force_codes':len(torch.unique(q[:,1])),'unique_code_pairs':len(unique),
          'largest_code_pair_fraction':counts.max().item()/len(q),'code_equivalent_top1':code_match,
          'random_candidate_code_match_expectation':chance,
          'pressure_last_frame_rmse_raw':torch.cat(pressure_sq).mean().sqrt().item(),
          'pressure_last_frame_mae_raw':torch.cat(pressure_mae).mean().item(),
          'force_last_frame_rmse_raw_per_axis':torch.cat(force_sq).mean(0).sqrt().tolist(),
          'train_mean_pressure_rmse_raw':torch.cat(base_ps).mean().sqrt().item(),
          'train_mean_force_rmse_raw_per_axis':torch.cat(base_fs).mean(0).sqrt().tolist(),
          'tactile_only_batch16_median_ms':float(np.median(timings)),
          'tactile_only_batch16_p10_p90_ms':np.quantile(timings,[.1,.9]).tolist(),
          'last_train':train[-1],'last_validation':val[-1]}
        (out/f'{name}.json').write_text(json.dumps(results[name],indent=2),encoding='utf-8')
        print(name,json.dumps({k:v for k,v in results[name].items() if k not in ['last_train','last_validation']}),flush=True)
        torch.save({'tactile':z,'force':c,'codes':q,'indices':indices},out/f'{name}_embeddings.pt')
        del model,ck,pred,x;gc.collect();torch.cuda.empty_cache()
    # Prove scaling is train-derived and identical across matched N values.
    scalar=np.sqrt(np.mean(norms['N5_cell']['pressure_std']**2))
    np.testing.assert_allclose(norms['N5_pooled']['pressure_std'],scalar,rtol=1e-6)
    np.testing.assert_array_equal(norms['N5_cell']['pressure_mean'],norms['N5_pooled']['pressure_mean'])
    for key in norms['N5_pooled']:
        np.testing.assert_array_equal(norms['N5_pooled'][key],norms['N1_pooled'][key])
    diagnostics={}
    for seq in sorted({e.sequence_id for e in dataset.episodes}):
        pressure=np.concatenate([dataset._sensor_rows(e,np.arange(e.length))[1] for e in dataset.episodes if e.sequence_id==seq])
        entry={}
        for mode in ['N5_cell','N5_pooled']:
            a=(pressure-norms[mode]['pressure_mean'])/norms[mode]['pressure_std']
            entry[mode]={'mean_square':float(np.mean(a*a)),'abs_p99':float(np.quantile(abs(a),.99)),'abs_max':float(abs(a).max())}
        diagnostics[seq]=entry
    report={'runs':results,'normalization_diagnostics':diagnostics,'pooled_pressure_scale':float(scalar),
            'checks':{'same_sample_indices':True,'same_endpoint_metadata':True,'same_splits':True,'train_only_pooled_formula':True,'same_N1_N5_normalization':True},
            'endpoints':endpoint_reference}
    (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    labels=list(results);fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].bar(labels,[results[k]['pressure_last_frame_rmse_raw'] for k in labels]);axes[0].axhline(results[labels[0]]['train_mean_pressure_rmse_raw'],color='black',linestyle='--',label='Train-mean predictor');axes[0].legend(fontsize=8);axes[0].set_title('Pressure-input reconstruction RMSE\nSame 256 validation endpoints, raw units')
    axes[1].bar(labels,[100*results[k]['exact_retrieval']['alignment/t2f/recall_at_1'] for k in labels]);axes[1].axhline(100/256,color='black',linestyle='--',label='Random ranking expectation');axes[1].legend(fontsize=8);axes[1].set_title('Exact tactile-to-force Recall@1 (%)\nSame 256 candidates; learned embeddings differ')
    axes[2].bar(labels,[results[k]['tactile_only_batch16_median_ms'] for k in labels]);axes[2].set_title('Tactile-only inference (ms / batch of 16)\nGPU, FP32, warmed, no data loading')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print('CONTROLLED COMPARISON CHECKS PASS',flush=True)
if __name__=='__main__':main()




