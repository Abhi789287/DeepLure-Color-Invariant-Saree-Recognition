import argparse, yaml, torch, numpy as np, pandas as pd
from PIL import Image
from .model import build_model
from .dataset import eval_transform
from .utils import device

@torch.no_grad()
def encode(model, paths, tfm, dev, batch=32):
    outs=[]
    for i in range(0,len(paths),batch):
        xs=torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]).to(dev)
        outs.append(model(xs).cpu())
    return torch.cat(outs)

def build_gallery(checkpoint, gallery_csv, config):
    dev=device(); ck=torch.load(checkpoint,map_location=dev)
    model=build_model(config["model"]["backbone"],config["model"]["embedding_dim"],False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    g=pd.read_csv(gallery_csv); z=encode(model,g.path.tolist(),eval_transform(config["data"]["image_size"]),dev)
    return model,g,z,dev

def identify(query_path, model, gallery, gallery_z, config, top_k=5, dev=None):
    dev=dev or device(); q=encode(model,[query_path],eval_transform(config["data"]["image_size"]),dev)[0]
    scores=(gallery_z @ q).numpy(); idx=np.argsort(-scores)[:top_k]
    out=gallery.iloc[idx].copy(); out["similarity"]=scores[idx]
    return out[["path","design_id","similarity"]]

def verify(path_a,path_b,model,config,threshold,dev=None):
    dev=dev or device(); z=encode(model,[path_a,path_b],eval_transform(config["data"]["image_size"]),dev)
    score=float(z[0]@z[1]); return score, score>=threshold

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/config.yaml"); ap.add_argument("--checkpoint",default="models/best_model.pth"); ap.add_argument("--query",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config)); model,g,z,dev=build_gallery(a.checkpoint,cfg["data"]["processed_dir"]+"/gallery.csv",cfg)
    print(identify(a.query,model,g,z,cfg,10,dev).to_string(index=False))
