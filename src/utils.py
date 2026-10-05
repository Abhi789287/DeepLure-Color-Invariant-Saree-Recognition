import os, random, numpy as np, torch

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
