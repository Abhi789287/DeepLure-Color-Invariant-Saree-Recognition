import torch
import torch.nn as nn

class BatchHardTripletLoss(nn.Module):
    """Select hardest positive and hardest negative for each anchor."""
    def __init__(self, margin=0.3): super().__init__(); self.margin=margin
    def forward(self, emb, labels):
        dist=1.0 - emb @ emb.t()
        labels=labels.view(-1,1)
        pos=(labels==labels.t()); neg=~pos
        eye=torch.eye(len(emb),device=emb.device,dtype=torch.bool)
        pos=pos & ~eye
        hardest_pos=dist.masked_fill(~pos,-1).max(dim=1).values
        hardest_neg=dist.masked_fill(~neg,1e9).min(dim=1).values
        valid=pos.any(dim=1) & neg.any(dim=1)
        if not valid.any(): return emb.sum()*0
        return torch.relu(hardest_pos[valid]-hardest_neg[valid]+self.margin).mean()
