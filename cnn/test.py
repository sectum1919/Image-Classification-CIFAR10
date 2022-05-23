import sys
sys.path.append("..")

import torch
from load_data import load_cifar10, load_cifar10_origin
from torchvision.transforms.functional import to_pil_image

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt

model = torch.load('./checkpoints/0.pt')
print(model)
cam_extractor = GradCAM(model, input_shape=(3, 32, 32))

trainset, trainloader, testset, testloader, classes = load_cifar10(batch_size=1)
originset, originloader = load_cifar10_origin(batch_size=1)


for iter, batch in enumerate(testloader):
    out = model(batch[0].cuda())
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    # Visualize the raw CAM

    origin_img = originset[iter][0]

    plt.imshow(to_pil_image(origin_img, mode='RGB'))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'/work9/cchen/project/study/ai/images/image_{iter}.png')

    plt.imshow(activation_map[0].squeeze(0).cpu().numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'/work9/cchen/project/study/ai/images/rawcam_{iter}.png')


    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(origin_img), to_pil_image(activation_map[0].squeeze(0).cpu(), mode='F'), alpha=0.5)
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'/work9/cchen/project/study/ai/images/overlay_{iter}.png')


    if iter==9:
        exit()