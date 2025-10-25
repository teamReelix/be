import torch
import torch.nn as nn
from .audio_cnn import SmallAudioCNN
from .video_cnn import SmallVideoCNN

class AVFusion(nn.Module):
    def __init__(self, vemb=256, aemb=128):
        super().__init__()
        self.v = SmallVideoCNN(emb=vemb)
        self.a = SmallAudioCNN(in_ch=1, emb=aemb)
        self.head = nn.Sequential(
            nn.Linear(vemb+aemb, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, xv, xa):
        hv = self.v(xv)
        ha = self.a(xa)
        h  = torch.cat([hv, ha], dim=1)
        return self.head(h).squeeze(1)