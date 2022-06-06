import sys
sys.path.append("..")

import torch
from load_data import load_cifar10, load_cifar10_origin
from torchvision.transforms.functional import to_pil_image

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt

from pathlib import Path

def generate_cam(model_path_list, layername_list, model_name_list, work_path='/work9/cchen/project/study/ai/images/'):
    
    trainset, trainloader, testset, testloader, classes = load_cifar10(batch_size=1)
    originset, originloader = load_cifar10_origin(batch_size=1)

    for i in range(len(model_path_list)):
        model_path = model_path_list[i]
        layername = layername_list[i]
        model_name = model_name_list[i]
        print(model_path)
        print(layername)
        print(model_name)

        image_path = f"{work_path}/{model_name}/"
        Path(image_path).mkdir(exist_ok=True, parents=True)

        model = torch.load(model_path, map_location=torch.device('cpu'))
        print(model)
        cam_extractor = GradCAM(model, input_shape=(3, 32, 32))


        for iter, batch in enumerate(testloader):
            # out = model(batch[0].cuda())
            out = model(batch[0])
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            # Visualize the raw CAM

            origin_img = originset[iter][0]

            plt.imshow(to_pil_image(origin_img, mode='RGB'))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{image_path}/image_{iter}.png', bbox_inches='tight')

            # plt.imshow(activation_map[0].squeeze(0).cpu().numpy())
            plt.imshow(activation_map[0].squeeze(0).numpy())
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{image_path}/rawcam_{iter}.png', bbox_inches='tight')


            # Resize the CAM and overlay it
            # result = overlay_mask(to_pil_image(origin_img), to_pil_image(activation_map[0].squeeze(0).cpu(), mode='F'), alpha=0.5)
            result = overlay_mask(to_pil_image(origin_img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            plt.imshow(result)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{image_path}/overlay_{iter}.png', bbox_inches='tight')


            if iter==9:
                break

generate_cam(
    [
        '/work9/cchen/project/study/ai/workpath/resnet18/checkpoints/199.pt',
        '/work9/cchen/project/study/ai/workpath/resnet34/checkpoints/199.pt',
        '/work9/cchen/project/study/ai/workpath/resnet50/checkpoints/199.pt',
        '/work9/cchen/project/study/ai/workpath/resnet101/checkpoints/199.pt',
        '/work9/cchen/project/study/ai/workpath/resnet152/checkpoints/199.pt',
    ],
    [
        'feature.6',
        'feature.6',
        'feature.6',
        'feature.6',
        'feature.6',
    ],
    [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
    ]
)