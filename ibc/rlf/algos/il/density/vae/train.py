import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib import MazeDataset as Dataset
from torch.utils.data import DataLoader
from lib import SimpleVAE
from lib import show
import torch
import argparse


def temp_show(ax, gt_samples, data, model, batch_size, fix=False):
    cur_ax = ax[0]
    cur_ax.clear()
    show(gt_samples, name='GT samples', ax=cur_ax)
    cur_ax = ax[1]
    cur_ax.clear()
    show(data, name='Sampled Z', ax=cur_ax)
    cur_ax = ax[2]
    cur_ax.clear()
    model.show(batch_size, ax=cur_ax)
    if fix:
        plt.show()
    else:
        plt.pause(0.002)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-load-path')
    args = parser.parse_args()
    env = args.traj_load_path.split('/')[-1].split('.')[0]

    data = torch.load(args.traj_load_path)
    obs = data['obs']
    actions = data['actions']
    dataset = torch.cat((obs, actions), 1)
    dataset = dataset.numpy()

     # visualize ground truth data
    data = Dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimpleVAE(in_channels=8, latent_dim=512)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1*1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=5000, gamma=.5)

    # training
    batch_size = 512 #512
    iter_num = 100000
    qbar = tqdm(total=iter_num)
    train_loss_list = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
    
    for iter in range(iter_num):

        optim.zero_grad()
        gt_samples = data.gen_data_xy(batch_size) # batch_size X 2
        tensor_gt_samples = torch.Tensor(gt_samples).to(device)
        forward_res = model(tensor_gt_samples) # [self.decode(z), input, mu, log_var, z] X batch_size

        loss = model.loss_function(forward_res, 0.001)
        loss['loss'].backward()

        optim.step()
        scheduler.step()
        train_loss_list.append(loss['loss'].detach().item())
        # model.show()
        #if iter % 200 == 0:
        #    temp_show(ax, gt_samples, forward_res[4].cpu().detach().numpy(), model, batch_size)

        if iter % 500 == 0:
            train_iteration_list = list(range(len(train_loss_list[100:])))
            plt.plot(train_iteration_list, train_loss_list[100:], color='r')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('maze2d_100' + '_vae_loss.png')
            plt.savefig('maze2d_100' + '_vae_loss_{}.png'.format(iter_num))
            plt.close()

        if iter % 1000 == 0:
            with torch.no_grad():
                # random sample images
                fig, axs = plt.subplots(1, 2, figsize=(28, 3))
                for idx, batch_x in tqdm(enumerate(dataloader)):
                    out = model(batch_x.to(device))
                    out = out[0].detach().cpu()

                    axs[0].scatter(batch_x[:, 0], batch_x[:, 1], color='blue', edgecolor='white')
                    axs[1].scatter(out[:, 0], out[:, 1], color='red', edgecolor='white')
                    file_name = 'maze2d' + '-vae-reconstruct-{}.png'.format(iter)
                plt.savefig(file_name)
                plt.close()  

            torch.save(model.state_dict(), 'maze2d' + '_vae_{}.pt'.format(iter_num))

        qbar.update(1)
        qbar.set_description(desc=f"step: {iter}, lr: {format(optim.param_groups[0]['lr'], '.2e')}, loss: {format(loss['loss'], '.3f')}, Reconstruction_Loss: {format(loss['Reconstruction_Loss'], '.3f')}, KLD: {format(loss['KLD'], '.3f')}.")

    pass