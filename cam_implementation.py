
#%%
import os
import PIL
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import torchvision

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

#%%
img_path = "cat.png"
pil_img = PIL.Image.open(img_path).convert('RGB')
width, height = pil_img.size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resized_torch_img = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()]
                                       )(pil_img).to(device)

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

#%%
model_squeezenet = models.squeezenet1_1(pretrained=True)



#%%
loaded_configs = [
    dict(model_type="alexnet", arch=model_alexnet, layer_name="feacture_11"),
    dict(model_type="vgg", arch=model_vgg, layer_name="features_29"),
    dict(model_type='resnet', arch=model_resnet, layer_name='layer4'),
    dict(model_type='densenet', arch=model_densenet, layer_name='features_norm5'),
    dict(model_type='squeezenet', arch=model_squeezenet, layer_name='features_12_expand3x3_activation')
]


#%%

for model_config in loaded_configs:
    model_config["arch"].to(device).eval()
    
cams = [[cls.from_config(**model_config) for cls in (GradCAM, GradCAMpp)] for model_config in loaded_configs]

#%%
images = []
for gradcam, gradcam_pp in cams:
    mask, _ = gradcam(normalized_torch_img)
    heatmap, result = visualize_cam(mask,resized_torch_img)
    
    mask_pp = gradcam_pp(normalized_torch_img)
    heatmap_pp, result_pp =visualize_cam(mask, resized_torch_img)

    images.extend([resized_torch_img.cpu(), result, result_pp])
    
grid_image = make_grid(images, nrow=3)

#%%

grid = transforms.ToPILImage()(grid_image)
grid



# %%
