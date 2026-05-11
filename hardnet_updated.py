import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class HardNet(nn.Module):
    def __init__(self):
        super(HardNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 8, bias=True)
        )

    def input_norm(self, x, eps=1e-7):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()

        b = x.size(0)
        flat = x.view(b, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)

        x = (x - mean.view(b, 1, 1, 1)) / (std.view(b, 1, 1, 1) + eps)
        return x

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()

        if x.ndim == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)

        x = self.input_norm(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.normalize(x, p=2, dim=1)

@torch.no_grad()
def extract_descriptors(self, image, keypoints, patch_size=32, device='cpu'):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    elif image.dim() == 3:
        image = image.unsqueeze(0) # (1, C, H, W)

    half_size = patch_size // 2
    patches = []

    for x, y in keypoints:
        x, y = int(x), int(y)
        patch = image[:, :, y-half_size:y+half_size, x-half_size:x+half_size]
        
        if patch.shape[2] != patch_size or patch.shape[3] != patch_size:
            patch = F.pad(patch, (0, patch_size - patch.shape[3], 0, patch_size - patch.shape[2]))
        
        patches.append(patch)

    if not patches:
        return torch.empty(0, 128)

    batch_patches = torch.cat(patches, dim=0).to(device)
    
    if batch_patches.max() > 1.0:
        batch_patches = batch_patches / 255.0
        
    return self.forward(batch_patches)

def load_hardnet_model(device='cuda'):
    model = HardNet()
    checkpoint = torch.load('./HardNetPS.pth', map_location=device)
    
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    
    state = {k.replace("module.", ""): v for k, v in state.items()}
    
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model