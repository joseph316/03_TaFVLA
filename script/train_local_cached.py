"""Run the normal trainer with an opt-in, pixel-verified local video cache."""
from pathlib import Path
import hydra
import torch
from script.local_rgb_cache import CachedTaFDataset
from script import train_adapter

torch.set_num_threads(4)
train_adapter.TaFDataset = CachedTaFDataset
@hydra.main(version_base=None, config_path=str(Path(__file__).resolve().parents[1]/'config'), config_name='taf_adapter')
def main(cfg):
    train_adapter.run_training(cfg)
if __name__ == '__main__':
    main()
