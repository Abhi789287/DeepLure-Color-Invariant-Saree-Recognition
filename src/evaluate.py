import argparse, yaml, torch, numpy as np, pandas as pd, json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from .model import build_model
from .dataset import eval_transform
from .inference import encode
from .utils import device, seed_everything

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/config.yaml"); ap.add_argument("--checkpoint",default="models/best_model.pth"); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config)); seed_everything(cfg["seed"]); dev=device()
    ck=torch.load(a.checkpoint,map_location=dev); model=build_model(cfg["model"]["backbone"],cfg["model"]["embedding_dim"],False).to(dev); model.load_state_dict(ck["model"]); model.eval()
    base=cfg["data"]["processed_dir"]; gal=pd.read_csv(base+"/gallery.csv"); qry=pd.read_csv(base+"/query.csv")
    tfm=eval_transform(cfg["data"]["image_size"]); gz=encode(model,gal.path.tolist(),tfm,dev); qz=encode(model,qry.path.tolist(),tfm,dev)
    scores=qz@gz.T
    top={}
    for k in cfg["evaluation"]["top_k"]:
        hits=0
        for i,row in qry.iterrows():
            inds=torch.topk(scores[i],min(k,len(gal))).indices.numpy()
            hits += int(row.design_id in set(gal.iloc[inds].design_id))
        top[f"top_{k}"]=hits/len(qry) if len(qry) else 0
    # Verification pairs from query/gallery with equal number of positives and negatives.
    pairs=[]; labels=[]
    for i,r in qry.iterrows():
        same=np.where(gal.design_id.values==r.design_id)[0]
        diff=np.where(gal.design_id.values!=r.design_id)[0]
        if len(same):
            j=np.random.choice(same); pairs.append(float(scores[i,j])); labels.append(1)
        if len(diff):
            j=np.random.choice(diff); pairs.append(float(scores[i,j])); labels.append(0)
    pairs=np.array(pairs); labels=np.array(labels)
    thresholds=np.linspace(-1,1,cfg["evaluation"]["threshold_grid_points"])
    best=(0,0)
    for t in thresholds:
        pred=(pairs>=t).astype(int); f1=precision_recall_fscore_support(labels,pred,average="binary",zero_division=0)[2]
        if f1>best[0]: best=(f1,t)
    pred=(pairs>=best[1]).astype(int)
    p,r,f,_=precision_recall_fscore_support(labels,pred,average="binary",zero_division=0)
    metrics={**top,"verification_accuracy":float(accuracy_score(labels,pred)),"precision":float(p),"recall":float(r),"f1":float(f),"roc_auc":float(roc_auc_score(labels,pairs)) if len(set(labels))==2 else None,"threshold":float(best[1]),"num_query":len(qry),"num_gallery":len(gal)}
    print(json.dumps(metrics,indent=2)); open("results/metrics.json","w").write(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
