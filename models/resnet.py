import torch
import torch.nn as nn
from torchvision.models import resnet34


class ResNet(nn.Module):
    def __init__(self, num_classes=4, dropout=.2):
        super().__init__()

        self.resnet = resnet34(weights='IMAGENET1K_V1')
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(128, num_classes)
    
    def forward(self, x):
        features = self.resnet(x.float())
        preds = self.final_layer(features)
        return preds
    
    def fine_tune(self, block_layer=5):
        for idx, child in enumerate(self.resnet.children()):
            if idx > block_layer:
                break
            for param in child.parameters():
                param.requires_grad = False