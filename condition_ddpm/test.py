import torchvision.transforms as transforms
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

tmp = torch.rand(3,256,256)
vutils.save_image(tmp, './test.png')


img2 = Image.open("./test.png").convert('RGB')
img2 = transform1(img2)
print(img2.size())