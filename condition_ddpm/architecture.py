import torch
from main import ContextUnet, DDPM

n_T = 400 # 500
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

print(ddpm)
