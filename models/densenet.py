import torch
import torch.nn as nn
from torchvision import models

class GlaucomaDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.classifier.in_features
        
        # Strip the original classification layer
        self.backbone.classifier = nn.Identity()
        
        # Your custom Glaucoma classification head
        self.head = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(num_ftrs, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(self.backbone(x))