import torch
from torch import nn

class BasiclBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        if in_channels!=out_channels:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut = nn.Sequential()
            
    def forward(self, x):
        return nn.ReLU(inplace=True)( self.conv(x) + self.shortcut(x) )

class StackedBasiclBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stack_counts) -> None:
        super().__init__()
        layers = []
        layers.append( BasiclBlock(in_channels, out_channels) )
        for _ in range(1,stack_counts,1):
            layers.append( BasiclBlock(out_channels, out_channels) )
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            StackedBasiclBlock(64, 64, 2),
            StackedBasiclBlock(64, 128, 2),
            StackedBasiclBlock(128, 256, 2),
            StackedBasiclBlock(256, 512, 2),

            nn.AvgPool2d(kernel_size=4)
        )
        
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet34(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            StackedBasiclBlock(64, 64, 3),
            StackedBasiclBlock(64, 128, 4),
            StackedBasiclBlock(128, 256, 6),
            StackedBasiclBlock(256, 512, 3),

            nn.AvgPool2d(kernel_size=4)
        )
        
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class BottleneckBlock(nn.Module):
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
            )
            self.shortcut = nn.Sequential()
    def forward(self, x):
        return nn.ReLU(inplace=True)( self.net(x) + self.shortcut(x) )

class StackedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, stack_counts) -> None:
        super().__init__()
        layers = []
        layers.append( BottleneckBlock(in_channels, hidden_channels, hidden_channels*4) )
        for _ in range(1,stack_counts,1):
            layers.append( BottleneckBlock(hidden_channels*4, hidden_channels, hidden_channels*4) )
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2),
            
            StackedBottleneckBlock(64, 64, 3),
            StackedBottleneckBlock(256, 128, 4),
            StackedBottleneckBlock(512, 256, 6),
            StackedBottleneckBlock(1024, 512, 3),
        
            nn.AvgPool2d(1, 1),
        )
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResNet101(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2),
            
            StackedBottleneckBlock(64, 64, 3),
            StackedBottleneckBlock(256, 128, 4),
            StackedBottleneckBlock(512, 256, 23),
            StackedBottleneckBlock(1024, 512, 3),
        
            nn.AvgPool2d(1, 1),
        )
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
        
class ResNet152(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2),
            
            StackedBottleneckBlock(64, 64, 3),
            StackedBottleneckBlock(256, 128, 8),
            StackedBottleneckBlock(512, 256, 36),
            StackedBottleneckBlock(1024, 512, 3),
        
            nn.AvgPool2d(1, 1),
        )
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)