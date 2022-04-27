from distutils.errors import LibError
import torch
from torch import nn
import math


def conv_block(in_channels, out_channels, kernel_list, padding_list):
    layers = []
    for i in range(0, len(kernel_list), 1):
        layers.append(nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_list[i],
                padding=padding_list[i],
            ))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


def vgg11():
    layers = []
    layers.extend( conv_block(3, 64, [3], [1]) )
    layers.extend( conv_block(64, 128, [3], [1]) )
    layers.extend( conv_block(128, 256, [3, 3], [1, 1]) )
    layers.extend( conv_block(256, 512, [3, 3], [1, 1]) )
    layers.extend( conv_block(512, 512, [3, 3], [1, 1]) )
    return nn.Sequential(*layers)

def vgg13():
    layers = []
    layers.extend( conv_block(3, 64, [3, 3], [1, 1]) )
    layers.extend( conv_block(64, 128, [3, 3], [1, 1]) )
    layers.extend( conv_block(128, 256, [3, 3], [1, 1]) )
    layers.extend( conv_block(256, 512, [3, 3], [1, 1]) )
    layers.extend( conv_block(512, 512, [3, 3], [1, 1]) )
    return nn.Sequential(*layers)

def vgg16():
    layers = []
    layers.extend( conv_block(3, 64, [3, 3], [1, 1]) )
    layers.extend( conv_block(64, 128, [3, 3], [1, 1]) )
    layers.extend( conv_block(128, 256, [3, 3, 3], [1, 1, 1]) )
    layers.extend( conv_block(256, 512, [3, 3, 3], [1, 1, 1]) )
    layers.extend( conv_block(512, 512, [3, 3, 3], [1, 1, 1]) )
    return nn.Sequential(*layers)

def vgg19():
    layers = []
    layers.extend( conv_block(3, 64, [3, 3], [1, 1]) )
    layers.extend( conv_block(64, 128, [3, 3], [1, 1]) )
    layers.extend( conv_block(128, 256, [3, 3, 3, 3], [1, 1, 1, 1]) )
    layers.extend( conv_block(256, 512, [3, 3, 3, 3], [1, 1, 1, 1]) )
    layers.extend( conv_block(512, 512, [3, 3, 3, 3], [1, 1, 1, 1]) )
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = vgg11()
        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
            # nn.Softmax(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        embedding = self.vgg(x)
        embedding = embedding.view(embedding.size(0), -1)
        output = self.linear(embedding)
        return output
