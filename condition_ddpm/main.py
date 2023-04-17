''' 
This script does conditional image generation on MNIST, using a diffusion model
This code is modified from,
https://github.com/cloneofsimo/minDiffusion
Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239
The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598
This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487
'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.datasets import make_s_curve, make_swiss_roll

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, n_steps, input_dim = 6, num_units = 256, n_classes=10):
        super(ContextUnet, self).__init__()
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim,num_units),
                nn.Linear(input_dim,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.Linear(input_dim,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.Linear(input_dim,num_units),
                nn.ReLU(),
                nn.Linear(num_units,input_dim),
            ]
        )

        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        )

        self.text_embeddings = nn.ModuleList(
            [
                nn.Embedding(2,num_units),
                nn.Embedding(2,num_units),
                nn.Embedding(3,num_units),
            ]
        )
        
    def forward(self, x, c, t):

        for idx, (step_embed, text_embed) in enumerate(zip(self.step_embeddings, self.text_embeddings)):
            print("** t:", t.size())
            print("** c:", c.size())
            step_embedding = step_embed(t)
            text_embedding = text_embed(c)
            x = self.linears[3*idx](x)
            x += step_embedding
            x = self.linears[3*idx+1](x)
            x += text_embeddin
            x = self.linears[3*idx+2](x)
            
        x = self.linears[-1](x)
        
        return x  


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    
    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = np.arange(10) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
        c_i = torch.from_numpy(c_i).to(device)

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)
            
            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


def train():
    
    # hardcoding these here
    n_epoch = 200
    n_T = 400 # 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 10
    n_feat =128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_param_path = "./params/"
    save_dir = './outputs10/'
    #ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(input_dim = 2, num_units = 256, n_classes=2), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(train_loader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            w ,label = next(iter(val_loader))
            n_sample = 10*n_classes
            w = w.to(device)
            label = label.to(device)

            x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=w)
        '''
            # append some real images at bottom, order by class also
            x_real = torch.Tensor(x_gen.shape).to(device)
            for k in range(n_classes):
                for j in range(int(n_sample/n_classes)):
                    try: 
                        idx = torch.squeeze((c == k).nonzero())[j]
                    except:
                        idx = 0
                    x_real[k+(j*n_classes)] = x[idx]
                
            grid = make_grid(x_gen*-1 + 1, nrow=10)
            save_image(grid, save_dir + f"image_ep{ep}.png")
            print('saved image at ' + save_dir + f"image_ep{ep}.png")
            
        # optionally save model
        if (ep+1)%5 == 0:
            torch.save(ddpm.state_dict(), save_param_path + f"model_{ep}.pth")
            print('saved model at ' + save_param_path + f"model_{ep}.pth")
        '''
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        data = self.data[item]
        state = data[0:2]
        goal = data[2].long()
        return state, goal

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    batch_size = 64
    
    # generate 10^5 points, get a s_curve
    swiss_curve, _ = make_swiss_roll(10**4,noise=0.1)
    swiss_curve = swiss_curve[:,[0,2]]/10.0
    s_curve, _ = make_s_curve(10**4,noise=0.1)
    s_curve = s_curve[:,[0,2]]/10.0

    s_data = torch.Tensor(s_curve).float()
    swiss_data = torch.Tensor(swiss_curve).float()

    # create dataset
    s_data = torch.concat((s_data, torch.zeros(10000,1)), 1)
    swiss_data = torch.concat((swiss_data, torch.ones(10000,1)), 1)
    dataset = torch.concat((s_data, swiss_data), 0)

    train_dataset, val_dataset = dataset[:8000], dataset[8000:]
    
    train_data = MyDataset(train_dataset)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_data = MyDataset(val_dataset)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train()