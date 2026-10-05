import argparse, yaml, time, torch, json
from pathlib import Path
from .model import build_model
from .utils import device

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def estimate_flops(model, x):
    """Uses torch.profiler operator FLOPs when available.
    This reports profiler-estimated FLOPs for supported operators (mainly conv/linear).
    """
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=False
        ) as prof:
            model(x)
        total = 0
        for evt in prof.key_averages():
            v = getattr(evt, "flops", 0)
            if v:
                total += v
        return int(total)
    except Exception as e:
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/config.yaml")
    ap.add_argument("--checkpoint",default="models/best_model.pth")
    a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config))
    dev=device()
    ck=torch.load(a.checkpoint,map_location=dev)
    m=build_model(cfg["model"]["backbone"],cfg["model"]["embedding_dim"],False).to(dev).eval()
    m.load_state_dict(ck["model"])
    size=cfg["data"]["image_size"]
    x=torch.randn(1,3,size,size,device=dev)

    # Warm-up and latency.
    with torch.no_grad():
        for _ in range(10): m(x)
        if dev.type=="cuda": torch.cuda.synchronize()
        t=time.perf_counter()
        for _ in range(50): m(x)
        if dev.type=="cuda": torch.cuda.synchronize()
        latency_ms=(time.perf_counter()-t)/50*1000

    flops=estimate_flops(m.cpu(), x.cpu())
    result={
        "parameters": parameter_count(m),
        "embedding_dim": cfg["model"]["embedding_dim"],
        "input_size": [1,3,size,size],
        "device_for_latency": str(dev),
        "latency_ms_per_image": latency_ms,
        "profiler_estimated_flops_per_image": flops,
        "note": "FLOPs are profiler estimates for supported operators; latency is measured empirically."
    }
    print(result)
    Path("results").mkdir(exist_ok=True)
    Path("results/efficiency.json").write_text(json.dumps(result,indent=2))

if __name__=="__main__": main()
