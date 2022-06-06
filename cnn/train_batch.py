import sys
sys.path.append("..")
import pickle
import torch.nn as nn
import torch.optim as optim
from vgg import VGG
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from senet import SEResNet50, SEResNet34
from trainer import train
from load_data import load_cifar10

def resnet_models(name):
    resnet_model_dict = {
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
    }
    return resnet_model_dict[name]()

model_name = sys.argv[1].lower()
if 'vgg' in model_name:
    model = VGG(model_name)
elif 'resnet' in model_name:
    model = resnet_models(model_name)
else:
    print("model not support")
    exit(-1)

model.cuda()

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
batch_size = 128
max_epoch = 200
print_per_iter = 50
trainset, trainloader, testset, testloader, classes = load_cifar10(batch_size)
train_loss, valid_loss, test_loss = train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    batch_size=batch_size,
    work_path=f'/work9/cchen/project/study/ai/workpath/{model_name}',
    max_epoch=max_epoch,
    print_per_iter=print_per_iter,
    train_loader=trainloader,
    valid_loader=testloader,
    test_loader=testloader,
)