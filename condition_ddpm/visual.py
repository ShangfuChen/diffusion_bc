from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image
import pandas as pd
from main import ContextUnet, DDPM

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
save_dir = './diffusion_outputs10/'

ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).to(device)

G_path = './params/model_199.pth'
ddpm.load_state_dict(torch.load(G_path))

ddpm.eval()
with torch.no_grad():
    num = batch_size // n_classes
    n_sample = num*n_classes

    x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=0.0)
    
    grid = make_grid(x_gen*-1 + 1, nrow=num)
    save_image(grid, save_dir + f"generate.png")
    print('saved image at ' + save_dir + f"generate.png")

for i in list([0, 4, 8, 12, 16, 26]):
    x_gen = x_gen_store[i]
    x_gen = x_gen*-1 + 1
    print(x_gen.shape)
    x_gen = torch.from_numpy(x_gen[0])
    save_image(x_gen, "./diffusion_outputs10/generate_t%d.png" %(400-i*20))
    print('saved image at ' + save_dir + f"generate_t{400-i*20}.png")    

