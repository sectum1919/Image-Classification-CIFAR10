import torch
from torch import nn

class SElayer(nn.Module):
    def __init__(self, in_channels, reduction=16) -> None:
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.pooling(x)
        y = y.view(b, c)
        y = self.net(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class BottleneckBlockSE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        if in_channels!=out_channels:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, 2),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1, 1),
                nn.BatchNorm2d(out_channels),
                SElayer(out_channels),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1, 1),
                nn.BatchNorm2d(out_channels),
                SElayer(out_channels),
            )
            self.shortcut = nn.Sequential()
    def forward(self, x):
        return nn.ReLU(inplace=True)( self.net(x) + self.shortcut(x) )

class StackedBottleneckBlockSE(nn.Module):
    def __init__(self, in_channels, hidden_channels, stack_counts) -> None:
        super().__init__()
        layers = []
        layers.append( BottleneckBlockSE(in_channels, hidden_channels, hidden_channels*4) )
        for _ in range(1,stack_counts,1):
            layers.append( BottleneckBlockSE(hidden_channels*4, hidden_channels, hidden_channels*4) )
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class SEResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            
            StackedBottleneckBlockSE(64, 64, 3),
            StackedBottleneckBlockSE(256, 128, 4),
            StackedBottleneckBlockSE(512, 256, 6),
            StackedBottleneckBlockSE(1024, 512, 3),
        
            nn.AvgPool2d(1, 1),
        )
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)