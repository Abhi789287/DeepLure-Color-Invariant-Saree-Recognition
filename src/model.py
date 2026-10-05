import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def build_model(name="resnet18", embedding_dim=256, pretrained=True):
    if name=="resnet18":
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        base=models.resnet18(weights=weights)
        feat=base.fc.in_features
        base.fc=nn.Identity()
    elif name=="resnet50":
        weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        base=models.resnet50(weights=weights)
        feat=base.fc.in_features
        base.fc=nn.Identity()
    else:
        raise ValueError("Supported backbones: resnet18, resnet50")
    head=nn.Sequential(nn.Linear(feat,512),nn.ReLU(inplace=True),nn.Dropout(.1),nn.Linear(512,embedding_dim))
    return EmbeddingNet(base,head)

class EmbeddingNet(nn.Module):
    def __init__(self, backbone, head):
        super().__init__(); self.backbone=backbone; self.head=head
    def forward(self,x):
        z=self.head(self.backbone(x))
        return F.normalize(z,p=2,dim=1)
