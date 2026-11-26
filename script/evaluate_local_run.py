"""Reload a locally produced checkpoint, evaluate fixed validation pairs, save figures."""
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from script.local_rgb_cache import CachedTaFDataset as TaFDataset
from utils.checkpoint import load_checkpoint
from utils.model_factory import build_model
from utils.alignment import evaluate_alignment, create_alignment_visualizations


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    ckpt = load_checkpoint(run / 'last.pt')
    cfg = ckpt['config']
    dc = cfg['data']
    dataset = TaFDataset(Path(dc['root_dir']).resolve(), window_size=dc['window_size'],
                         stride=dc['stride'], image_size=tuple(dc['image_size']),
                         sensor_types=dc['sensor_types'], max_sequences=dc['max_sequences'],
                         timestamp_cache_enabled=True, show_initialization_progress=True)
    indices = ckpt['alignment']['sample_indices']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    assert all(torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point())
    loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=False, num_workers=0)
    model.eval()
    with torch.inference_mode():
        batch = next(iter(loader))
        inputs = {k: batch[k].to(device) for k in ['tactile','force','pressure']}
        tactile_only = model.encode_tactile(inputs['tactile'])
        paired = model(**inputs)['z_tactile']
        torch.testing.assert_close(tactile_only, paired)
        print('Tactile-only inference PASS: no force/pressure input required', flush=True)
    result = evaluate_alignment(model, loader, device)
    out = run / 'evaluation'
    out.mkdir(exist_ok=True)
    _, figures = create_alignment_visualizations(result, max_pairs=64, random_seed=42,
        title_label=f"N=5, step {ckpt['step']}, held-out sequence (256-pair retrieval)",
        file_prefix='final', output_dir=out, save_local_figures=True)
    torch.save({'tactile': result.tactile, 'force': result.force, 'indices': indices}, out / 'embeddings.pt')
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    train = [r for r in records if 'train/loss_total' in r]
    val = [r for r in records if 'val/loss_total' in r]
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for rows, label in [(train,'Train (rolling mean)'), (val,'Validation (160 windows)')]:
        axes[0].plot([r['step'] for r in rows],[r[f"{'train' if rows is train else 'val'}/loss_total"] for r in rows],label=label)
    axes[0].set_yscale('log'); axes[0].set_title('Total loss (log scale)'); axes[0].legend(fontsize=8)
    for key in ['loss_nce','loss_recon','loss_quant']:
        axes[1].plot([r['step'] for r in train],[r['train/'+key] for r in train],label=key)
    axes[1].set_title('Training loss components'); axes[1].legend(fontsize=8)
    for branch in ['pressure','force']:
        axes[2].plot([r['step'] for r in train],[r['train/'+branch+'_codebook_perplexity'] for r in train],label=branch)
    axes[2].set_title('Per-batch codebook perplexity (batch=16)'); axes[2].legend()
    for ax in axes: ax.set_xlabel('Step'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=160); plt.close(fig)
    import numpy as np
    diagnostics = {}
    for seq in sorted({e.sequence_id for e in dataset.episodes}):
        values = [dataset._sensor_rows(e, np.arange(e.length))[1] for e in dataset.episodes if e.sequence_id == seq]
        pressure = np.concatenate(values)
        mean = ckpt['normalization']['pressure_mean'].numpy()
        std = ckpt['normalization']['pressure_std'].numpy()
        normalized = (pressure-mean)/np.maximum(std,1e-6)
        diagnostics[seq] = {'raw_pressure_min':float(pressure.min()),'raw_pressure_max':float(pressure.max()),
            'normalized_abs_quantiles':np.quantile(np.abs(normalized),[.5,.9,.99,1]).tolist(),
            'mean_normalized_pressure_square':float(np.mean(normalized**2))}
    diagnostics['training_pressure_std_quantiles'] = np.quantile(std,[0,.1,.5,.9,1]).tolist()
    optimizer_steps = sorted({int(v['step'].item()) for v in ckpt['optimizer']['state'].values() if 'step' in v})
    summary={'optimizer_update_counts':optimizer_steps,'pressure_diagnostics':diagnostics,'step':ckpt['step'],'split':ckpt['split'],'metrics':result.metrics,
             'first_train':train[0],'last_train':train[-1],'first_validation':val[0],
             'last_validation':val[-1],'figures':figures,
             'logged_nonfinite_grad_steps':[r['step'] for r in train if not __import__('math').isfinite(r['grad_norm'])]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()





