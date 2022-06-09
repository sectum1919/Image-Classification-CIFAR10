import sys

sys.path.append("..")

import pickle

import torch.nn as nn
import torch.optim as optim

from vit_model import vit_patch
from trainer import train

from load_data import load_cifar10


def test_ViT(patch_size, depth, num_heads):
    model = vit_patch(patch_size, depth, num_heads)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    batch_size = 128
    max_epoch = 500
    print_per_iter = 50
    trainset, trainloader, testset, testloader, classes = load_cifar10(batch_size)

    train_loss, valid_loss, test_loss = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epoch=max_epoch,
        print_per_iter=print_per_iter,
        train_loader=trainloader,
        valid_loader=testloader,
        test_loader=testloader,
    )




# test_seresnet()
# test_vgg()
test_ViT(patch_size = 4, depth = 12, num_heads = 4)
