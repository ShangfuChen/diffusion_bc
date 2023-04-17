from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image
import pandas as pd
from main import ContextUnet, DDPM
import os
import sys

output_dir = sys.argv[1]
print("output_dir:", output_dir)
    
# Configure data loader
batch_size = 100

class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader
        
    def __getitem__(self, item):
        img_name, label = self.data[item]
        img = self.loader(img_name)
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)
    
def train_loader(file):
    return Image.open(train_path+file).convert('RGB')
    #return mpimg.imread(path+file)

def val_loader(file):
    return Image.open(valid_path+file).convert('RGB')
    #return mpimg.imread(path+file)
    
transform = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

# hardcoding these here
n_T = 400 # 500
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = output_dir

ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).to(device)

for k in range(10):
    #G_path = 'problem2/params/model_199.pth'
    G_path = 'model_199.pth'
    ddpm.load_state_dict(torch.load(G_path))

    torch.manual_seed(k)
    np.random.seed(k)

    ddpm.eval()
    with torch.no_grad():
        num = batch_size // n_classes
        n_sample = num*n_classes

        x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=2.0)

        for i in range(n_classes):
            for j in range(batch_size//n_classes):
                tmp = x_gen[i*10+j]
                tmp = tmp*-1 + 1
                vutils.save_image(tmp, os.path.join(output_dir, "%d_%03d.png" %(i, k*10+j+1)))
                print('saved image at ' + output_dir + f"%d_%03d.png" %(i, k*10+j+1)) 
