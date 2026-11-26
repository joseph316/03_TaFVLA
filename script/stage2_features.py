"""Frozen FP32 backbone features with numerical equivalence checks and provenance."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
import timm
from script.local_rgb_cache import CachedTaFDataset

BACKBONE='vit_small_patch16_224.augreg_in21k_ft_in1k'
FEATURES=Path('../cache/backbone_fp32').resolve()

def cache_name(seq):return FEATURES/f'{seq}.npy'

def main():
    torch.set_num_threads(4);FEATURES.mkdir(parents=True,exist_ok=True)
    backbone=timm.create_model(BACKBONE,pretrained=True,num_classes=0).cuda().eval().requires_grad_(False)
    digest=hashlib.sha256()
    for k,v in backbone.state_dict().items():digest.update(k.encode());digest.update(v.cpu().numpy().tobytes())
    weights_sha=digest.hexdigest();cfg=backbone.pretrained_cfg
    mean=torch.tensor(cfg['mean'],device='cuda').view(1,3,1,1);std=torch.tensor(cfg['std'],device='cuda').view(1,3,1,1)
    for root in ['../data/TaF-Dataset','../data/TaF-Holdout']:
        ds=CachedTaFDataset(Path(root).resolve(),window_size=10,stride=5,image_size=(224,224))
        seen=set()
        for ep in ds.episodes:
            if ep.sequence_id in seen:continue
            seen.add(ep.sequence_id);out=cache_name(ep.sequence_id);meta=out.with_suffix('.json')
            source={'backbone':BACKBONE,'weights_sha256':weights_sha,'video':str(ep.video.resolve()),'video_size':ep.video.stat().st_size,'video_mtime_ns':ep.video.stat().st_mtime_ns,'tensor_layout':'contiguous_NCHW','precision':'float32','dimensions':384,'resize':'PyAV AREA 224x224'}
            frames,pts=ds._rgb_maps[str(ep.video.resolve())];source['frames']=len(frames)
            if not (out.exists() and meta.exists() and json.loads(meta.read_text())['source']==source):
                print(f'Extracting {ep.sequence_id}: {len(frames)} frames',flush=True)
                array=np.lib.format.open_memmap(out,mode='w+',dtype=np.float32,shape=(len(frames),384))
                with torch.inference_mode():
                    for start in range(0,len(frames),64):
                        x=torch.from_numpy(np.array(frames[start:start+64])).permute(0,3,1,2).contiguous().float().cuda()/255
                        array[start:start+len(x)]=backbone((x-mean)/std).cpu().numpy()
                        if start%6400==0:print(f'{ep.sequence_id} {start}/{len(frames)}',flush=True)
                array.flush();del array
                # Validate with a different batch shape against direct image inference.
                cached=np.load(out,mmap_mode='r');check=np.linspace(0,len(frames)-1,10,dtype=int)
                with torch.inference_mode():
                    x=torch.from_numpy(np.array(frames[check])).permute(0,3,1,2).contiguous().float().cuda()/255
                    direct=backbone((x-mean)/std).cpu()
                torch.testing.assert_close(direct,torch.from_numpy(np.array(cached[check])),rtol=2e-4,atol=2e-5)
                meta.write_text(json.dumps({'source':source,'equivalence_max_abs_error':float((direct-torch.from_numpy(np.array(cached[check]))).abs().max()),'check_frame_indices':check.tolist()},indent=2),encoding='utf-8')
                print(f'Feature equivalence PASS {ep.sequence_id}',flush=True)
    print('ALL FEATURE CACHES READY',flush=True)
if __name__=='__main__':main()
