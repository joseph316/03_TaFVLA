"""N-frame alignment training on frozen, verified raw ViT features; common Nmax=10 support."""
from pathlib import Path
import json,types
import numpy as np
import torch
import hydra
from dataset.taf_dataset import TaFDataset
from script import train_adapter
from utils.model_factory import build_model as original_build
from script.stage2_features import cache_name

class FeatureDataset(TaFDataset):
    def __init__(self,*args,window_size=5,**kwargs):
        self.requested_window=int(window_size)
        if self.requested_window not in (1,3,5,7,10):raise ValueError('Unsupported N')
        super().__init__(*args,window_size=10,**kwargs)
        self.features={s:np.load(cache_name(s),mmap_mode='r') for s in {e.sequence_id for e in self.episodes}}
        for e in self.episodes:
            metadata=json.loads(cache_name(e.sequence_id).with_suffix('.json').read_text())['source']
            assert metadata['video']==str(e.video.resolve()) and metadata['video_size']==e.video.stat().st_size
            assert metadata['video_mtime_ns']==e.video.stat().st_mtime_ns
        self._sensor_maps={}
    def _sensor_rows(self,ep,offsets):
        self._sensor_cache=self._sensor_maps.get(str(ep.parquet))
        answer=super()._sensor_rows(ep,offsets)
        self._sensor_maps[str(ep.parquet)]=self._sensor_cache
        return answer
    def __getitem__(self,index):
        ep,start=self._locate(index)
        offsets=np.arange(start+10-self.requested_window,start+10,dtype=np.int64)
        force,pressure=self._sensor_rows(ep,offsets)
        features=np.array(self.features[ep.sequence_id][ep.video_frame_indices[offsets]])
        return {'tactile':torch.from_numpy(features),'force':torch.from_numpy(force),'pressure':torch.from_numpy(pressure),
            'metadata':{'sensor_type':ep.sensor_type,'sequence_id':ep.sequence_id,'episode_index':ep.episode_index,
              'frame_indices':torch.from_numpy(ep.frames[offsets].copy()),'timestamps':torch.from_numpy(ep.timestamps[offsets].copy()),
              'video_timestamps':torch.from_numpy(ep.video_timestamps[offsets].copy()),'video_frame_indices':torch.from_numpy(ep.video_frame_indices[offsets].copy())}}

def feature_forward(self,tactile,force,pressure):
    if tactile.shape[1]!=self.window_size:raise ValueError('Wrong feature history length')
    projected=self.tactile_encoder.projection(tactile)
    return self.forward_from_tactile_features(projected,force,pressure)

def build_feature_model(cfg):
    model=original_build(cfg)
    if not model.tactile_encoder.freeze_backbone:raise ValueError('Caching requires frozen backbone')
    model.forward=types.MethodType(feature_forward,model)
    return model

torch.set_num_threads(4)
@hydra.main(version_base='1.3',config_path=str(Path(__file__).resolve().parents[1]/'config'),config_name='taf_adapter')
def main(cfg):
    train_adapter.TaFDataset=FeatureDataset
    train_adapter.build_model=build_feature_model
    train_adapter.run_training(cfg)
if __name__=='__main__':main()
