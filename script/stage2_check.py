from pathlib import Path
import numpy as np
import torch
from script.stage2_train import FeatureDataset,feature_forward
from script.local_rgb_cache import CachedTaFDataset
from utils.checkpoint import load_checkpoint
from utils.model_factory import build_model

def main():
    torch.set_num_threads(4)
    ds=FeatureDataset(Path('../data/TaF-Dataset').resolve(),window_size=5,stride=5,image_size=(224,224))
    raw=CachedTaFDataset(Path('../data/TaF-Dataset').resolve(),window_size=10,stride=5,image_size=(224,224))
    ranges=[];previous=0
    for end in ds._ends:ranges.extend([previous,end-1]);previous=end
    for index in ranges:
        ep,start=ds._locate(index)
        for n in [1,3,5,7,10]:
            ds.requested_window=n;sample=ds[index]
            assert sample['tactile'].shape==(n,384)
            assert int(sample['metadata']['frame_indices'][-1])==int(ep.frames[start+9])
            assert torch.isfinite(sample['tactile']).all()
    ds.requested_window=5
    ck=load_checkpoint(Path('../runs/controlled_20261113_114543/N5_cell/last.pt'))
    model=build_model(ck['config'],pretrained=False).cuda().eval();model.load_state_dict(ck['model'])
    a=ds[0];b=raw[0];images=b['tactile'][-5:].unsqueeze(0).cuda();force=a['force'].unsqueeze(0).cuda();pressure=a['pressure'].unsqueeze(0).cuda();features=a['tactile'].unsqueeze(0).cuda()
    direct=model(images,force,pressure);direct['loss_total'].backward();gradient=model.tactile_encoder.projection.weight.grad.clone();model.zero_grad(set_to_none=True)
    cached=feature_forward(model,features,force,pressure);cached['loss_total'].backward()
    torch.testing.assert_close(direct['z_tactile'],cached['z_tactile'],rtol=2e-4,atol=3e-5)
    torch.testing.assert_close(gradient,model.tactile_encoder.projection.weight.grad,rtol=3e-3,atol=5e-5)
    print('PASS: 60 episode-boundary anchors x 5 N values; direct/cached embeddings and trainable projection gradients equivalent')
if __name__=='__main__':main()
