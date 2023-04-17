#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve, make_swiss_roll
import torch
import cv2
import os, sys
import argparse
import numpy as np
import scipy.stats
from geomloss import SamplesLoss
import gym
import d4rl # Import required to register environments
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler
from tqdm import tqdm

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

########### hyper parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128 #128
num_epoch = 4000
num_steps = 100

# decide beta
betas = torch.linspace(-6,6,num_steps).to(device)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

# calculate alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt
alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0)
alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device),alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape
print("all the same shape",betas.shape)

########### decide the sample during definite diffusion process
# calculate x on given time based on x_0 and re-parameterization
def q_x(x_0,t):
    """based on x[0], get x[t] on any given time t"""
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise) # adding noise based on x[0]在x[0]

########### gaussian distribution in reverse diffusion process
import torch
import torch.nn as nn

class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, input_dim=6, num_units=128):
        super(MLPDiffusion,self).__init__()
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim,num_units),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.Linear(num_units,num_units),
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
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        )
        
    def forward(self, x, c, t):

        for idx, (step_embed, text_embed) in enumerate(zip(self.step_embeddings, self.text_embeddings)):
            step_embed = step_embed(t)
            text_embed = text_embed(c)
            x = self.linears[3*idx](x)
            x += step_embed
            x = self.linears[3*idx+1](x)
            x += text_embed
            x = self.linears[3*idx+2](x)
            
        x = self.linears[-1](x)
        
        return x   
    
########### training loss funciton
# sample at any given time t, and calculate sampling loss
def diffusion_loss_fn(model, x_0, c, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, device='cuda'):
    batch_size = x_0.shape[0]
    
    # generate eandom t for a batch data
    t = torch.randint(0, n_steps, size=(batch_size//2,)).to(device)
    t = torch.cat([t, n_steps-1-t], dim=0) #[batch_size, 1]
    t = t.unsqueeze(-1)
    
    # coefficient of x0
    a = alphas_bar_sqrt[t]
    
    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]
    
    # generate random noise eps
    e = torch.randn_like(x_0)
    
    # model input
    x = x_0*a + e*aml
    
    # get predicted randome noise at time t
    output = model(x,c,t.squeeze(-1))
    
    # calculate the loss between actual noise and predicted noise
    return (e - output).square().mean()

########### reverse diffusion sample function（inference）
def p_sample_loop(model, shape, c, n_steps, betas, one_minus_alphas_bar_sqrt,device='cuda'):
    # generate[T-1]、x[T-2]|...x[0] from x[T]
    cur_x = torch.randn(shape).to(device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,c,i,betas,one_minus_alphas_bar_sqrt,device)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model, x, c, t, betas,one_minus_alphas_bar_sqrt,device='cuda'):
    # sample reconstruction data at time t drom x[T]
    t = torch.tensor([t]).to(device)

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x,c,t)
 
    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
    
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
   
    sample = mean + sigma_t * z
   
    return (sample)

########### start training, print loss and print the medium reconstrction result
seed = 1234

class EMA(): # Exponential Moving Average
    #EMA
    def __init__(self,mu=0.01):
        self.mu = mu
        self.shadow = {}
        
    def register(self,name,val):
        self.shadow[name] = val.clone()
        
    def __call__(self,name,x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0-self.mu)*self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

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

if __name__ == '__main__':
    
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

    sample_num = dataset.size()[0]
    if  sample_num % 2 == 1:
        dataset = dataset[1:sample_num, :]
    print("after", dataset.size())

    train_dataset, val_dataset = dataset[:8000], dataset[8000:]
    
    train_data = MyDataset(train_dataset)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_data = MyDataset(val_dataset)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print('Training model...')

    # output dimension is 8，inputs are x and step
    model = MLPDiffusion(num_steps, input_dim=dataset.shape[1]-1, num_units=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    train_loss_list = []
    val_loss_list = []
    for t in tqdm(range(1,num_epoch+1)):
        model.train()
        for idx, (batch_x, goal) in enumerate(train_loader):
            batch_x, goal = batch_x.to(device), goal.to(device)
            batch_x = batch_x.squeeze()
            loss = diffusion_loss_fn(model,batch_x,goal,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
            train_loss_list.append(loss.cpu().detach().item())
            #print("loss.cpu():", loss.cpu().detach().item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for idx, (batch_x, goal) in enumerate(train_loader):
                batch_x, goal = batch_x.to(device), goal.to(device)
                batch_x = batch_x.squeeze()
                loss = diffusion_loss_fn(model,batch_x,goal,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
                val_loss_list.append(loss.cpu().detach().item())
    
    x_seq = p_sample_loop(model, (10000,2), torch.LongTensor([0]).to(device), num_steps, betas, one_minus_alphas_bar_sqrt)

    fig, axs = plt.subplots(1, 10, figsize=(28, 3))
    for i in range(1, 11):
        cur_x = x_seq[i * 10].detach().cpu()
        axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white')
        axs[i - 1].set_axis_off()
        axs[i - 1].set_title('$q(\mathbf{x}_{' + str(i * 10) + '})$')

    #env = args.traj_load_path.split('/')[-1].split('.')[0]
    torch.save(model.state_dict(), 'test_ddpm_nextobs.pt')
    #torch.save(model.state_dict(), env + '_ddpm_nextobs.pt')

    iteration_list = list(range(len(train_loss_list)))
    plt.plot(iteration_list, train_loss_list, color='r')
    plt.plot(iteration_list, val_loss_list, color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('test_ddpm_loss.png')
    plt.savefig('test_ddpm_loss_.png')
    #plt.title(env + '_model_loss.png')
    #plt.savefig(env + '_model_loss_.png')
    plt.close()
