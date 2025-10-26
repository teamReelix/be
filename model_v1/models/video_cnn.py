import torch
import torch.nn as nn

# from config import NUM_FRAMES

NUM_FRAMES = 8

class SmallVideoCNN(nn.Module):
    def __init__(self, emb=256):
        super().__init__()
        in_ch = 3*NUM_FRAMES
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),  nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(256, emb)
    def forward(self, x):  # (B,3,T,H,W)
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).contiguous().view(B, C*T, H, W)
        h = self.net(x).flatten(1)
        return self.fc(h)