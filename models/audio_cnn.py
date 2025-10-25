import torch
import torch.nn as nn

class SmallAudioCNN(nn.Module):
    def __init__(self, in_ch=1, emb=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),    nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),   nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, emb)
    def forward(self, x):  # (B,1,M,T)
        h = self.net(x).flatten(1)
        return self.fc(h)