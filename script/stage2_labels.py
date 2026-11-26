"""Audit fixed upstream centered labels; do not tune thresholds using held-out results."""
import argparse,hashlib,json,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path('../dataset_visualization-main/src').resolve()))
from taf_contact_explorer.config import load_config
from taf_contact_explorer.analysis.preprocessing import preprocess_signals
from taf_contact_explorer.analysis.centered_mtcc import classify_centered_mtcc_episode
from taf_contact_explorer.analysis.mtcc import classify_mtcc_episode
from dataset.taf_dataset import TaFDataset

PHASES=['No Contact','Loading','Holding','Unloading','Transition / Unknown']
COLORS=['#999999','#e66101','#1b9e77','#7570b3','#e6ab02']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='../data/TaF-Dataset');parser.add_argument('--out',required=True);args=parser.parse_args()
    torch.set_num_threads(4);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cp=Path('../dataset_visualization-main/configs/classifier.yaml');cfg=load_config(cp)
    (out/'classifier.yaml').write_bytes(cp.read_bytes())
    dataset=TaFDataset(Path(args.root).resolve(),window_size=10,stride=5,image_size=(224,224))
    parts=[];segments=[];episodes=[]
    for ep in dataset.episodes:
        force,pressure=dataset._sensor_rows(ep,np.arange(ep.length))
        sig=preprocess_signals(ep.timestamps,force,pressure,cfg.preprocessing)
        centered=classify_centered_mtcc_episode(sig,cfg.centered_mtcc,cfg.classification)
        causal=classify_mtcc_episode(sig,cfg.mtcc,cfg.classification)
        assert len(centered.labels)==ep.length and len(causal)==ep.length
        labels=centered.labels['contact_phase'].astype(str).to_numpy();other=causal['contact_phase'].astype(str).to_numpy()
        assert set(labels)<=set(PHASES)
        # Verify four-class rules on every stored segment; Unknown is explicitly retained.
        for row in centered.segments.itertuples():
            phase=row.assigned_contact_phase
            if phase=='Loading':assert row.beta_segment>=cfg.centered_mtcc.positive_slope_threshold and row.Delta_F_segment>=cfg.centered_mtcc.positive_net_change_threshold
            if phase=='Unloading':assert row.beta_segment<=cfg.centered_mtcc.negative_slope_threshold and row.Delta_F_segment<=cfg.centered_mtcc.negative_net_change_threshold
            if phase=='Holding':assert abs(row.beta_segment)<=cfg.centered_mtcc.holding_slope_max and abs(row.Delta_F_segment)<=cfg.centered_mtcc.holding_net_change_max
        table=pd.DataFrame({'sequence':ep.sequence_id,'episode':ep.episode_index,'frame':ep.frames,'timestamp':ep.timestamps,'phase':labels,'causal_phase':other,'normal_force':sig.normal_force,'pressure_total':pressure.reshape(ep.length,-1).sum(1),'segment':centered.labels['segment_id'].to_numpy()})
        parts.append(table);seg=centered.segments.copy();seg['sequence']=ep.sequence_id;seg['episode']=ep.episode_index;segments.append(seg)
        episodes.append({'sequence':ep.sequence_id,'episode':ep.episode_index,'frames':ep.length,'auto_sign':sig.applied_sign,'episode_baseline':sig.baseline,'centered_causal_agreement':float(np.mean(labels==other)),'counts':dict(zip(*np.unique(labels,return_counts=True)))})
        # Representative audit: first episode of each sequence, plus all raw tables.
        if ep.episode_index==0:
            fig,axs=plt.subplots(3,1,figsize=(14,7),sharex=True)
            axs[0].plot(ep.timestamps,sig.normal_force,lw=.8);axs[0].set_ylabel('Calibrated Fz');axs[0].set_title(f'{ep.sequence_id} episode {ep.episode_index}: heuristic labels, not ground truth')
            axs[1].plot(ep.timestamps,pressure.reshape(ep.length,-1).sum(1),lw=.8);axs[1].set_ylabel('Pressure sum')
            for label,offset in [(labels,0),(other,1.2)]:
                for i,phase in enumerate(PHASES):
                    mask=label==phase;axs[2].scatter(ep.timestamps[mask],np.full(mask.sum(),offset),s=6,color=COLORS[i],label=phase if offset==0 else None)
            axs[2].set_yticks([0,1.2],['Centered','Trailing*']);axs[2].set_xlabel('Episode time (s)');axs[2].legend(ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.3))
            fig.tight_layout();fig.savefig(out/f'{ep.sequence_id}_episode0.png',dpi=140,bbox_inches='tight');plt.close(fig)
        print(f'Labelled {ep.sequence_id} ep{ep.episode_index}: {dict(zip(*np.unique(labels,return_counts=True)))}',flush=True)
    all_labels=pd.concat(parts,ignore_index=True);all_segments=pd.concat(segments,ignore_index=True)
    all_labels.to_parquet(out/'labels.parquet',index=False);all_segments.to_parquet(out/'segments.parquet',index=False)
    counts=all_labels.groupby(['sequence','phase']).size().unstack(fill_value=0)
    counts.to_csv(out/'phase_counts.csv')
    confusion=pd.crosstab(all_labels['phase'],all_labels['causal_phase']);confusion.to_csv(out/'classifier_agreement.csv')
    report={'config_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'config':asdict(cfg),'episodes':episodes,'row_count':len(all_labels),'overall_classifier_agreement':float((all_labels.phase==all_labels.causal_phase).mean()),'rule_checks':'PASS','human_ground_truth_available':False,'causality_note':'Trailing classifier uses episode-wide auto-sign/low-pressure baseline and centered Savitzky-Golay preprocessing; end-to-end causal claim is invalid with these defaults. Both outputs used only for offline analysis.'}
    def encode(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    (out/'audit.json').write_text(json.dumps(report,indent=2,default=encode),encoding='utf-8')
    # Export balanced manual-review candidates without assigning fabricated human labels.
    review=[]
    for phase in PHASES:
        candidates=all_segments[(all_segments.assigned_contact_phase==phase)&(all_segments.duration_seconds>=.5)]
        for _,r in candidates.sort_values('duration_seconds',ascending=False).head(6).iterrows():
            review.append({'sequence':r.sequence,'episode':r.episode,'start_frame':r.start_frame,'end_frame':r.end_frame,'suggested_phase':phase,'human_phase':'','boundary_error_seconds':'','review_note':''})
    pd.DataFrame(review).to_csv(out/'human_review_candidates.csv',index=False)
    print(counts.to_string());print('LABEL RULE AUDIT PASS; human confirmation remains open',flush=True)
if __name__=='__main__':main()
