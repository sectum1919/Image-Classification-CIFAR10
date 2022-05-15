import sys

sys.path.append("..")

import pickle

import torch.nn as nn
import torch.optim as optim

from vgg import VGG
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from trainer import train

from load_data import load_cifar10


def test_resnet():
    model = ResNet34()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    batch_size = 256
    max_epoch = 200
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




def test_vgg():
    model = VGG()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # optimizer = optim.Adam(model.parameters(), lr=1e-2)
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=5e-4, momentum=0.9)
    batch_size = 128
    max_epoch = 100
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

    pickle.dump({"train_loss":train_loss, "valid_loss":valid_loss, "test_loss":test_loss}, open("./train.res",'w'))

test_resnet()