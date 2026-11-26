from pathlib import Path
import hashlib,json,numpy as np,torch,timm
from script.stage2_train import FeatureDataset
from script.local_rgb_cache import CachedTaFDataset
from utils.checkpoint import load_checkpoint
from utils.model_factory import build_model
from script.stage2_features import BACKBONE

torch.set_num_threads(4)
ck=load_checkpoint(Path('../runs/controlled_20261113_114543/N5_cell/last.pt'));m=build_model(ck['config'],pretrained=False).cuda().eval();m.load_state_dict(ck['model'])
h=hashlib.sha256()
for k,v in m.tactile_encoder.backbone.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
print('CHECKPOINT BACKBONE SHA',h.hexdigest());print(Path('../cache/backbone_fp32/gs_mini_obj1.json').read_text())
ds=FeatureDataset(Path('../data/TaF-Dataset').resolve(),window_size=5,stride=5,image_size=(224,224));raw=CachedTaFDataset(Path('../data/TaF-Dataset').resolve(),window_size=10,stride=5,image_size=(224,224));a=ds[0];b=raw[0];x=b['tactile'][-5:].unsqueeze(0).cuda();features=a['tactile'].unsqueeze(0).cuda()
print('PIXEL NORMALIZATION',m.tactile_encoder.pixel_mean.flatten(),m.tactile_encoder.pixel_std.flatten())
print('PRETRAINED CONFIG',m.tactile_encoder.backbone.pretrained_cfg.get('mean'),m.tactile_encoder.backbone.pretrained_cfg.get('std'))
with torch.no_grad():
 direct=m.tactile_encoder.backbone(((x-m.tactile_encoder.pixel_mean)/m.tactile_encoder.pixel_std).reshape(5,3,224,224))
 print('BACKBONE MAX DIFFERENCE',float((direct-features[0]).abs().max()))
with torch.no_grad():
 xx=((x-m.tactile_encoder.pixel_mean)/m.tactile_encoder.pixel_std).reshape(5,3,224,224)
 alternate=m.tactile_encoder.backbone(xx.contiguous(memory_format=torch.channels_last))
 print('CHANNELS LAST difference from cache',float((alternate-features[0]).abs().max()))
