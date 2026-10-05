import argparse, os, random
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import yaml

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp"}

def scan(root):
    rows=[]
    root=Path(root)
    for cls in sorted([p for p in root.iterdir() if p.is_dir()]):
        for p in cls.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                try:
                    with Image.open(p) as im: im.verify()
                    rows.append({"path":str(p),"design_id":cls.name})
                except Exception as e:
                    print("Skipping unreadable:", p, e)
    return pd.DataFrame(rows)

def make_splits(df, seed, val_frac, gallery_frac, query_frac):
    rng=random.Random(seed); parts={"train":[],"val":[],"gallery":[],"query":[]}
    for label, g in df.groupby("design_id"):
        items=g["path"].tolist(); rng.shuffle(items)
        n=len(items)
        if n < 4:
            # Keep small classes in train; they cannot safely support all evaluation partitions.
            parts["train"] += items
            continue
        nq=max(1, round(n*query_frac)); ng=max(1, round(n*gallery_frac))
        nv=max(1, round(n*val_frac))
        if nq+ng+nv >= n: nv=max(1,n-nq-ng-1)
        q=items[:nq]; ga=items[nq:nq+ng]; va=items[nq+ng:nq+ng+nv]; tr=items[nq+ng+nv:]
        parts["query"] += q; parts["gallery"] += ga; parts["val"] += va; parts["train"] += tr
    out={}
    label_map={x:i for i,x in enumerate(sorted(df.design_id.unique()))}
    for k,paths in parts.items():
        d=df[df.path.isin(paths)].copy()
        d["label"]=d.design_id.map(label_map)
        out[k]=d[["path","design_id","label"]].reset_index(drop=True)
    return out, label_map

class SareeDataset(Dataset):
    def __init__(self, csv, transform=None):
        self.df=pd.read_csv(csv) if isinstance(csv,(str,Path)) else csv.reset_index(drop=True)
        self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        im=Image.open(r.path).convert("RGB")
        if self.transform: im=self.transform(im)
        return im, int(r.label), r.design_id, r.path

def train_transform(size):
    return transforms.Compose([
        transforms.Resize((size+24,size+24)),
        transforms.RandomResizedCrop(size, scale=(0.70,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.70, hue=.08),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])

def eval_transform(size):
    return transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])

class BalancedClassSampler(Sampler):
    def __init__(self, labels, images_per_class=4):
        self.labels=list(labels); self.k=images_per_class
        self.classes=sorted(set(self.labels))
        self.by={c:[i for i,x in enumerate(self.labels) if x==c] for c in self.classes}
    def __iter__(self):
        pools={c:v[:] for c,v in self.by.items()}
        for v in pools.values(): random.shuffle(v)
        batch=[]
        for c in self.classes:
            if len(pools[c]) < self.k: continue
            batch.extend(pools[c][:self.k])
        random.shuffle(batch)
        return iter(batch)
    def __len__(self): return sum(min(len(v),self.k) for v in self.by.values())

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/config.yaml"); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config)); seed=cfg["seed"]
    df=scan(cfg["data"]["raw_dir"])
    if df.empty: raise SystemExit("No images found. Put images under data/raw/<design_id>/...")
    out,_=make_splits(df,seed,cfg["data"]["val_fraction"],cfg["data"]["gallery_fraction"],cfg["data"]["query_fraction"])
    d=Path(cfg["data"]["processed_dir"]); d.mkdir(parents=True,exist_ok=True)
    for k,v in out.items(): v.to_csv(d/f"{k}.csv",index=False); print(k,len(v), "designs",v.design_id.nunique())
