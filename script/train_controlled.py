"""Controlled pilot: same five-frame support/endpoints for N=1 and N=5."""
import os
from pathlib import Path
import hydra
import torch
from script.local_rgb_cache import CachedTaFDataset
from script import train_adapter
from utils.data import compute_normalization as original_normalization

class MatchedDataset(CachedTaFDataset):
    def __init__(self,*args,window_size=5,**kwargs):
        self.requested_window = int(window_size)
        if self.requested_window not in (1,5): raise ValueError('Pilot supports N=1 or N=5')
        super().__init__(*args,window_size=5,**kwargs)
    def __getitem__(self,index):
        sample=super().__getitem__(index)
        if self.requested_window == 1:
            for key in ['tactile','force','pressure']:
                sample[key]=sample[key][-1:]
            for key,value in sample['metadata'].items():
                if isinstance(value,torch.Tensor) and value.ndim and len(value)==5:
                    sample['metadata'][key]=value[-1:]
        return sample

def normalization(dataset,train_episode_ids,chunk_frames,eps):
    stats=original_normalization(dataset,train_episode_ids,chunk_frames,eps)
    mode=os.environ.get('TAF_PRESSURE_SCALING','cell')
    if mode == 'pooled':
        scale=stats['pressure_std'].square().mean().sqrt().clamp_min(eps)
        stats['pressure_std']=torch.full_like(stats['pressure_std'],scale)
        print(f'Train-only pooled within-cell pressure SD: {scale.item():.8f}',flush=True)
    elif mode != 'cell': raise ValueError(mode)
    return stats

torch.set_num_threads(4)
@hydra.main(version_base='1.3',config_path=str(Path(__file__).resolve().parents[1]/'config'),config_name='taf_adapter')
def main(cfg):
    train_adapter.TaFDataset=MatchedDataset
    train_adapter.compute_normalization=normalization
    train_adapter.run_training(cfg)
if __name__=='__main__':main()
