import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from script.local_rgb_cache import CachedTaFDataset
from utils.checkpoint import load_checkpoint
from utils.model_factory import build_model

def main():
 p=argparse.ArgumentParser();p.add_argument('group',type=Path);a=p.parse_args();g=a.group.resolve();torch.set_num_threads(4)
 ds=CachedTaFDataset(Path('../data/TaF-Dataset').resolve(),window_size=10,stride=5,image_size=(224,224))
 images=torch.stack([ds[i]['tactile'] for i in range(16)]);results={}
 for n in [1,3,5,7,10]:
  ck=load_checkpoint(g/f'N{n}_seed42/last.pt');m=build_model(ck['config'],pretrained=False).cuda().eval();m.load_state_dict(ck['model']);rows={}
  for b in [1,16]:
   x=images[:b,-n:].cuda()
   with torch.inference_mode():
    for _ in range(10):m.encode_tactile(x)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(30):
     start=time.perf_counter();m.encode_tactile(x);torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
   rows[str(b)]={'median_ms':float(np.median(times)),'p10_p90_ms':np.quantile(times,[.1,.9]).tolist(),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
   del x
  results[str(n)]=rows;print(n,rows,flush=True);del m,ck;gc.collect();torch.cuda.empty_cache()
 (g/'probes/latency.json').write_text(json.dumps({'device':torch.cuda.get_device_name(),'precision':'FP32','includes':'image->frozen ViT->trained temporal encoder; excludes loader, transfer, ridge probe and contact-state decision','warmup':10,'repetitions':30,'results':results},indent=2),encoding='utf-8')
if __name__=='__main__':main()
