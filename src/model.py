from torchvision import models
import torch.nn as nn
import torch 
from torchsummary import summary

class PoseNetG(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        
        self.backbone.fc = nn.Sequential(*[
            nn.Linear(1024, 2048), 
            nn.ReLU(), 
            nn.Linear(2048, 7)
        ])
        
    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        return self._normalize_pose(out)

    def _normalize_pose(self, pose: torch.Tensor):
        pos = pose[:, :3]
        quat = pose[:, 3:]
        quat = quat / quat.norm(p=2, dim=1, keepdim=True)
        return torch.cat([pos, quat], dim=1)

class PoseNetR34(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        self.backbone.fc = nn.Sequential(*[
            nn.Linear(512, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 7)
        ])
    
    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        return self._normalize_pose(out)

    def _normalize_pose(self, pose: torch.Tensor):
        pos = pose[:, :3]
        quat = pose[:, 3:]
        quat = quat / quat.norm(p=2, dim=1, keepdim=True)
        return torch.cat([pos, quat], dim=1)

class PoseNetR18(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.backbone.fc = nn.Sequential(*[
            nn.Linear(512, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 7)
        ])
    
    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        return self._normalize_pose(out)

    def _normalize_pose(self, pose: torch.Tensor):
        pos = pose[:, :3]
        quat = pose[:, 3:]
        quat = quat / quat.norm(p=2, dim=1, keepdim=True)
        return torch.cat([pos, quat], dim=1)

class PoseNetEffB0(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        self.backbone.classifier = nn.Sequential(*[
            nn.Linear(1280, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 7)
        ])
    
    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        return self._normalize_pose(out)

    def _normalize_pose(self, pose: torch.Tensor):
        pos = pose[:, :3]
        quat = pose[:, 3:]
        quat = quat / quat.norm(p=2, dim=1, keepdim=True)
        return torch.cat([pos, quat], dim=1)

class PoseNetEffB1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        
        self.backbone.classifier = nn.Sequential(*[
            nn.Linear(1280, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 7)
        ])
    
    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        return self._normalize_pose(out)

    def _normalize_pose(self, pose: torch.Tensor):
        pos = pose[:, :3]
        quat = pose[:, 3:]
        quat = quat / quat.norm(p=2, dim=1, keepdim=True)
        return torch.cat([pos, quat], dim=1)

class PoseNetMobV3L(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        self.backbone.classifier = nn.Sequential(*[
            nn.Linear(960, 512), 
            nn.ReLU(),
            nn.Linear(512, 7)
        ])
    
    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        return self._normalize_pose(out)

    def _normalize_pose(self, pose: torch.Tensor):
        pos = pose[:, :3]
        quat = pose[:, 3:]
        quat = quat / quat.norm(p=2, dim=1, keepdim=True)
        return torch.cat([pos, quat], dim=1)

class PoseNetMobV3S(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        self.backbone.classifier = nn.Sequential(*[
            nn.Linear(576, 512), 
            nn.ReLU(),
            nn.Linear(512, 7)
        ])
    
    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        return self._normalize_pose(out)

    def _normalize_pose(self, pose: torch.Tensor):
        pos = pose[:, :3]
        quat = pose[:, 3:]
        quat = quat / quat.norm(p=2, dim=1, keepdim=True)
        return torch.cat([pos, quat], dim=1)