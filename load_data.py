import torch
import torchvision
import torchvision.transforms as transforms

from skimage import feature as ft
from tqdm import tqdm


def load_cifar10(batch_size=64):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="/work9/cchen/project/study/ai/data",
        train=True,
        download=True,
        transform=train_transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root="/work9/cchen/project/study/ai/data",
        train=False,
        download=True,
        transform=test_transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainset, trainloader, testset, testloader, classes


def extract_hog(data_list):
    print("extracting hog features")
    hog_list = [ ft.hog(img, channel_axis=2) for img in tqdm(data_list) ]
    return hog_list
