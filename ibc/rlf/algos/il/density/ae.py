import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
import numpy as np
import torchvision.utils as vutils
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_s_curve, make_swiss_roll
import matplotlib.pyplot as plt
import os, sys
import argparse

epochs = 4000
batch_size = 64
use_cuda = 1
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")

# Model structure
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 64),
            nn.Sigmoid(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Linear(16, 5),
        )
    def forward(self, inputs):
        x = inputs
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx % 2 == 1:
                x = x*7
        codes = x
        return codes

class Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Decoder, self).__init__()
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(5, 256),
            nn.Sigmoid(),
            nn.Linear(256, 64),
            nn.Sigmoid(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Linear(16, input_dim),
            nn.Sigmoid(),
        )
    def forward(self, inputs):
        x = inputs
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            if idx % 2 == 1:
                x = x*7
        outputs = x
        return outputs

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.input_dim = input_dim
        self.encoder = Encoder(self.input_dim)
        # Decoder
        self.decoder = Decoder(self.input_dim)
        self.loss_function = nn.MSELoss()

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        return self.decoder(inputs)

    def get_loss(self, inputs):
        with torch.no_grad():
            codes = self.encoder(inputs)
            decoded = self.decoder(codes)
            loss = self.loss_function(decoded, inputs).detach().cpu()  
        return loss

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        data = self.data[item]
        state = data
        #goal = data[2].long()
        return state, state

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    # Create the environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-load-path')
    args = parser.parse_args()
    
    data = torch.load(args.traj_load_path)
    obs = data['obs']
    actions = data['actions']
    next_obs = data['next_obs']
    
    dataset = torch.cat((obs, actions), 1)
    sample_num = dataset.size()[0]
    if  sample_num % 2 == 1:
        dataset = dataset[1:sample_num, :]
    print("after", dataset.size())
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

    train_dataset, val_dataset = dataset[:8000], dataset[8000:]
    
    train_data = MyDataset(train_dataset)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_data = MyDataset(val_dataset)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=0)
 
    dim = dataset.size()[1]
    model_ae = AutoEncoder(input_dim=dim).to(device)
    model_ae.load_state_dict(torch.load('AutoEncoder_maze2d_4000.pth'))
    #model_ae = torch.load('AutoEncoder_maze2d_000.pth')
    optimizer = torch.optim.SGD(model_ae.parameters(), lr=5*1e-5)
    loss_function = nn.MSELoss().to(device)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,40], gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience=2, verbose = True)

    # Train
    log_loss=[]
    for epoch in range(0, epochs):
    #for epoch in range(epochs):
        total_loss = 0
        for data, _ in dataloader:
            inputs = data.to(device) 
            model_ae.zero_grad()
            # Forward
            codes, decoded = model_ae(inputs)
            loss = loss_function(decoded, inputs)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach()
            total_loss+=loss
        ave_loss = total_loss / len(dataloader.dataset)
        #scheduler.step(ave_loss)
        log_loss.append(ave_loss)
        
        if epoch % 5 ==0:
            print('[{}/{}] Loss:'.format(epoch, epochs), ave_loss.item())
    print('[{}/{}] Loss:'.format(epoch, epochs), ave_loss.item())

    plt.plot(log_loss)
    plt.savefig('AE_loss.png')
    plt.close()
    torch.save(model_ae.state_dict(), 'AutoEncoder_maze2d_{}.pth'.format(epoch+1))
    #torch.save(model_ae, 'AutoEncoder_maze2d_4000.pth')

        # inference
    with torch.no_grad():
        # random sample images
        z = torch.randn(18524, 5).to(device)
        out = model_ae.decode(z).cpu()
        #print("out:", out.size())
        plt.scatter(out[:, 0], out[:, 1], color='red', edgecolor='white')
        plt.savefig('results/curve-sample-{}.png'.format(epoch+1))
        plt.close()

        # reconstruct images
        fig, axs = plt.subplots(1, 2, figsize=(28, 3))
        x = dataset.to(device)
        _, out = model_ae(x)
        x, out = x.cpu().detach(), out.cpu().detach()
        axs[0].scatter(x[:, 0], x[:, 1], color='blue', edgecolor='white')
        axs[1].scatter(out[:, 0], out[:, 1], color='red', edgecolor='white')
        #axs[1].set_yticks((-2,-1,0,1,2))
        plt.savefig('results/curve-recons-{}.png'.format(epoch+1))
        plt.close()
