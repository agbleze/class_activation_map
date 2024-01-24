
#%%
import nltk
nltk.download("wordnet")

#%%
from nbdt.model import SoftNBDT, HardNBDT
from pytorchcv.models.wrn_cifar import wrn28_10_cifar10
from torchvision import transforms

from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet

import sys

maybe_install_wordnet()

#%%
def load_model():
    model = wrn28_10_cifar10()
    model = HardNBDT(
        pretrained=True, dataset="CIFAR10", arch="wrn28_10_cifar10",
        model=model
    )
    return model


def load_image():
    assert len(sys.argv) > 1
    im = load_image_from_path("https://bsmedia.business-standard.com/media-handler.php?mediaPath=https://bsmedia.business-standard.com/_media/bs/img/article/2019-12/23/full/1577083902-3265.jpg&width=1200")
    transform = transforms.Compose([
      transforms.Resize(32),
      transforms.CenterCrop(32),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    x = transform(im)[None]
    return x


def heirerchy_output(outputs, decisions):
    _, predicted = outputs.max(1)
    predicted_class = DATASET_TO_CLASSES['CIFAR10'][predicted[0]]
    print("Predicted class:", predicted_class,
          "\n\nHeirerch:",
          ', '.join(['\n{} ({:.2f}%)'.format(info['name'], info['prob'] * 100) 
          for info in decisions[0]][1:]))
    
def main():
    model = load_model()
    x = load_image()
    outputs, decisions = model.forward_with_decisions(x)
    heirerchy_output(outputs, decisions)


#%%    
if __name__ == "__main__":
    main()



# %%
