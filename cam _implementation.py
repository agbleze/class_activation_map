
import os
import PIL
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import torchvision

from gradcam.utils import visualuize_cam
from gradcam import GradCAM, GradCAMpp


img_path = ""
pil_img = PIL.Image.open(img_path)
width, height = pil_img.size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resized_torch_img = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])(pil_img).to(device)

normalized_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(resized_torch_img)[None]

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                     )
])

normalized_img = transform(pil_img)

model_alexnet = models.alexnet(pretrained=True)
model_vgg = models.vgg16(pretrained=True)
model_resnet = models.resnet101(pretrained=True)
model_densenet = models.densenet161(pretrained=True)
model_squeezenet = models.squeezenet1_1(pretrained=True)

