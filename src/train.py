import argparse, os, yaml, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import seed_everything, device
from .dataset import SareeDataset, train_transform, BalancedClassSampler
from .model import build_model
from .losses import BatchHardTripletLoss

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/config.yaml"); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config)); seed_everything(cfg["seed"]); dev=device()
    trcsv=cfg["data"]["processed_dir"]+"/train.csv"
    ds=SareeDataset(trcsv,train_transform(cfg["data"]["image_size"]))
    loader=DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=True,num_workers=cfg["training"]["num_workers"],pin_memory=True)
    model=build_model(cfg["model"]["backbone"],cfg["model"]["embedding_dim"],cfg["model"]["pretrained"]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["lr"],weight_decay=cfg["training"]["weight_decay"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg["training"]["epochs"])
    loss_fn=BatchHardTripletLoss(cfg["training"]["margin"])
    scaler=torch.amp.GradScaler("cuda",enabled=cfg["training"]["amp"] and dev.type=="cuda")
    best=1e9; os.makedirs("models",exist_ok=True)
    for ep in range(cfg["training"]["epochs"]):
        model.train(); total=0
        for x,y,_,_ in tqdm(loader,desc=f"epoch {ep+1}"):
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=(dev.type=="cuda")):
                z=model(x); loss=loss_fn(z,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()
        sched.step(); avg=total/max(1,len(loader)); print("loss",avg)
        if avg<best:
            best=avg; torch.save({"model":model.state_dict(),"config":cfg,"epoch":ep+1},"models/best_model.pth")
    print("saved models/best_model.pth")

if __name__=="__main__": main()
