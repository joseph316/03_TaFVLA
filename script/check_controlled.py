"""Check endpoint matching at every episode boundary before temporal comparisons."""
from pathlib import Path
import torch
from script.train_controlled import MatchedDataset
from utils.data import episode_window_ranges

def main():
    ds=MatchedDataset(Path('../data/TaF-Dataset').resolve(),window_size=5,stride=5,image_size=(224,224))
    checked=0
    for indices in episode_window_ranges(ds):
        for index in [indices.start,indices.stop-1]:
            ds.requested_window=5;full=ds[index]
            ds.requested_window=1;single=ds[index]
            for key in ['tactile','force','pressure']:
                torch.testing.assert_close(single[key],full[key][-1:],rtol=0,atol=0)
            for key in ['frame_indices','timestamps','video_timestamps','video_frame_indices','video_pts']:
                torch.testing.assert_close(single['metadata'][key],full['metadata'][key][-1:],rtol=0,atol=0)
            assert single['metadata']['episode_index']==full['metadata']['episode_index']
            checked+=1
    print(f'PASS: {checked} boundary windows; N=1 pixels, sensors, timestamps equal N=5 last frame')
if __name__=='__main__':main()
