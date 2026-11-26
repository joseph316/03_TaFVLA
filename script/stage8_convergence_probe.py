"""Compare 1k/3k/5k checkpoints on development targets only; no test access."""
from pathlib import Path
import json,gc
import numpy as np
import pandas as pd
import torch
from utils.checkpoint import load_checkpoint
from utils.model_factory import build_model
from script.stage2_probe import fit_probe

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/convergence_N3_seed42_20261124';BASE=ROOT/'runs/stage2_20261120_115709/probes/dev_bundle'
def main():
    torch.set_num_threads(4);out=RUN/'dev_probes';out.mkdir(exist_ok=True)
    x=np.load(BASE/'features.npy',mmap_mode='r');y=np.load(BASE/'targets.npy');meta=pd.read_parquet(BASE/'metadata.parquet');train=meta.sequence.isin(['gs_mini_obj1','gs_mini_obj3']).to_numpy();val=(meta.sequence=='gs_mini_obj2').to_numpy();assert (train|val).all()
    mean=y[train].mean(0);scale=y[train].std(0);scale[:6]=np.maximum(scale[:6],1e-6);scale[6:]=max(float(np.sqrt(np.mean(scale[6:]**2))),1e-6);rows=[]
    for step in [1000,3000,5000]:
        ck=load_checkpoint(RUN/f'step_{step:08d}.pt');assert ck['step']==step
        model=build_model(ck['config'],pretrained=False).cuda().eval();model.load_state_dict(ck['model'],strict=True);parts=[]
        with torch.inference_mode():
            for start in range(0,len(x),256):
                raw=torch.from_numpy(np.array(x[start:start+256,-3:])).cuda();parts.append(model.tactile_encoder.aggregate_features(model.tactile_encoder.projection(raw)).cpu().numpy())
        del model,ck;gc.collect();torch.cuda.empty_cache();z=np.concatenate(parts)
        record,_=fit_probe(z,y,train,val,mean,scale,meta,out/f'step_{step}_probe.npz');rows.append({'step':step,'alpha':record['alpha'],**record['metrics']['ALL']});print(rows[-1],flush=True)
    (out/'results.json').write_text(json.dumps({'scope':'development only; exploratory single-seed convergence pilot; no heldout evaluation','results':rows},indent=2));pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
if __name__=='__main__':main()
