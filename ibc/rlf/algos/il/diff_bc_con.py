import os
import sys

import copy
import gym
import numpy as np
import rlf.algos.utils as autils
import rlf.rl.utils as rutils
import torch
import torch.nn.functional as F
from rlf.algos.il.base_il import BaseILAlgo
from rlf.args import str2bool
from rlf.storage.base_storage import BaseStorage
from tqdm import tqdm
from rlf.algos.il.ddpm import MLPDiffusion
import math

class Diff_bc(BaseILAlgo):
    """
    When used as a standalone updater, BC will perform a single update per call
    to update. The total number of udpates is # epochs * # batches in expert
    dataset. num-steps must be 0 and num-procs 1 as no experience should be collected in the
    environment. To see the performance, you must evaluate. To just evaluate at
    the end of training, set eval-interval to a large number that is greater
    than the number of updates. There will always be a final evaluation.
    """

    def __init__(self, set_arg_defs=True):
        super().__init__()
        self.set_arg_defs = set_arg_defs

    def init(self, policy, envs, args):
        super().init(policy, args)
        self.envs = envs
        self.num_epochs = 0
        self.action_dim = rutils.get_ac_dim(self.policy.action_space)
        if self.args.bc_state_norm:
            self.norm_mean = self.expert_stats["state"][0]
            self.norm_var = torch.pow(self.expert_stats["state"][1], 2)
        else:
            self.norm_mean = None
            self.norm_var = None
        self.num_bc_updates = 0
       
    # sample at any given time t, and calculate sampling loss
    def diffusion_loss_fn(self, model, x_0_pred, x_0_expert, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        batch_size = x_0_pred.shape[0]
        
        # generate eandom t for a batch data
        t = torch.randint(0,n_steps,size=(batch_size//2,))
        t = torch.cat([t,n_steps-1-t],dim=0) #[batch_size, 1]
        t = t.unsqueeze(-1)
        
        # coefficient of x0
        a = alphas_bar_sqrt[t].to(self.args.device)
        
        # coefficient of eps
        aml = one_minus_alphas_bar_sqrt[t].to(self.args.device)
        
        # generate random noise eps
        e = torch.randn_like(x_0_pred).to(self.args.device)
        e2 = torch.randn_like(x_0_expert).to(self.args.device)
        
        # model input
        x = x_0_pred*a + e*aml
        x2 = x_0_expert*a + e2*aml
        
        # get predicted randome noise at time t
        output = model(x, t.squeeze(-1).to(self.args.device))
        output2 = model(x2, t.squeeze(-1).to(self.args.device))
        
        # calculate the loss between actual noise and predicted noise
        loss = (e - output).square().mean()
        loss2 = (e2 - output2).square().mean()
        
        return loss, loss2


    def get_density(self, states, pred_next, pred_action, expert_next, expert_action):
        num_steps = 100
        dim = states.size()[1]*2 + pred_action.size()[1] - 2
        model = MLPDiffusion(num_steps, input_dim = dim, device = self.args.device)
        current_directory = os.path.dirname(os.path.abspath(__file__))
        #weight_path = os.path.join(current_directory, 'ddpm.pt')
        weight_path = self.args.ddpm_path
        model.load_state_dict(torch.load(weight_path))
        # decide beta
        betas = torch.linspace(-6,6,num_steps)
        betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5
        betas = betas.to(self.args.device)
        # calculate alpha��alpha_prod��alpha_prod_previous��alpha_bar_sqrt
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas,0).to(self.args.device)
        alphas_prod_p = torch.cat([torch.tensor([1]).float().to(self.args.device),alphas_prod[:-1]],0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        num_steps = 100
        
        pred = torch.cat((states, pred_next[:,:4], pred_action), 1)
        expert = torch.cat((states, expert_next[:,:4], expert_action), 1)
        pred_loss, expert_loss = self.diffusion_loss_fn(model, pred, expert, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        
        return pred_loss, expert_loss
             

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if args.bc_state_norm:
            print("Setting environment state normalization")
            settings.state_fn = self._norm_state
        return settings

    def _norm_state(self, x):
        obs_x = torch.clamp(
            (rutils.get_def_obs(x) - self.norm_mean)
            / torch.pow(self.norm_var + 1e-8, 0.5),
            -10.0,
            10.0,
        )
        if isinstance(x, dict):
            x["observation"] = obs_x
        return x

    def get_num_updates(self):
        if self.exp_generator is None:
            return len(self.expert_train_loader) * self.args.bc_num_epochs
        else:
            return self.args.exp_gen_num_trans * self.args.bc_num_epochs

    def get_completed_update_steps(self, num_updates):
        return num_updates * self.args.traj_batch_size

    def _reset_data_fetcher(self):
        super()._reset_data_fetcher()
        self.num_epochs += 1

    def full_train(self, update_iter=0):
        action_loss = []
        diff_loss = []
        prev_num = 0

        # First BC
        with tqdm(total=self.args.bc_num_epochs) as pbar:
            while self.num_epochs < self.args.bc_num_epochs:
                super().pre_update(self.num_bc_updates)
                log_vals = self._bc_step(False)
                action_loss.append(log_vals["_pr_action_loss"])
                diff_loss.append(log_vals["_pr_diff_loss"])

                pbar.update(self.num_epochs - prev_num)
                prev_num = self.num_epochs

        rutils.plot_line(
            action_loss,
            f"action_loss_{update_iter}.png",
            self.args.vid_dir,
            not self.args.no_wb,
            self.get_completed_update_steps(self.update_i),
        )
        self.num_epochs = 0

    def pre_update(self, cur_update):
        # Override the learning rate decay
        pass

    def _bc_step(self, decay_lr):
        if decay_lr:
            super().pre_update(self.num_bc_updates)
        expert_batch = self._get_next_data()
        if expert_batch is None:
            self._reset_data_fetcher()
            expert_batch = self._get_next_data()

        states, true_next_states, true_actions = self._get_data(expert_batch)

        log_dict = {}
        pred_actions, _, _ = self.policy(states, None, None)
        pred_next_states = torch.zeros_like(states)
        for i, (state, action) in  enumerate( zip(states, pred_actions)):
            self.envs.reset_model(state)
            pred_next_state, _, _, _ = self.envs.step(action.detach())
            pred_next_states[i] = pred_next_state

        #obs = self.envs.reset()
        #print("*** state[0]:", states[0])
        #print("*** obs:", obs)
        
        if rutils.is_discrete(self.policy.action_space):
            pred_label = rutils.get_ac_compact(self.policy.action_space, pred_actions)
            acc = (pred_label == true_actions.long()).sum().float() / pred_label.shape[
                0
            ]
            log_dict["_pr_acc"] = acc.item()
        loss = autils.compute_ac_loss(
            pred_actions,
            true_actions.view(-1, self.action_dim),
            self.policy.action_space,
        )
        
        pred_loss, expert_loss = self.get_density(states, pred_next_states, 
                pred_actions, true_next_states, true_actions)
        pred_density, expert_density = math.exp(-pred_loss), math.exp(-expert_loss)
        diff_loss = pred_loss - expert_loss
        
        total_loss = loss + diff_loss

        self._standard_step(total_loss) #backward
        self.num_bc_updates += 1
        
        #print("************************")
        #print("pred_loss:", pred_loss.item())
        #print("expert_loss:", expert_loss.item())
        #print("pred_density:", pred_density)
        #print("expert_density:", expert_density)

        val_loss = self._compute_val_loss()
        if val_loss is not None:
            log_dict["_pr_val_loss"] = val_loss.item()

        log_dict["_pr_action_loss"] = loss.item()
        log_dict["_pr_diff_loss"] = diff_loss.item()
        
        return log_dict

    def _get_data(self, batch):
        states = batch["state"].to(self.args.device)
        next_states = batch["next_state"].to(self.args.device)
        if self.args.bc_state_norm:
            states = self._norm_state(states)
            next_states = self._norm_state(next_states)

        if self.args.bc_noise is not None:
            add_noise = torch.randn(states.shape) * self.args.bc_noise
            states += add_noise.to(self.args.device)
            states = states.detach()

        true_actions = batch["actions"].to(self.args.device)
        true_actions = self._adjust_action(true_actions)
        return states, next_states, true_actions

    def _compute_val_loss(self):
        if self.update_i % self.args.eval_interval != 0:
            return None
        if self.val_train_loader is None:
            return None
        with torch.no_grad():
            action_losses = []
            diff_losses = []
            for batch in self.val_train_loader:
                states, true_next_states, true_actions = self._get_data(batch)
                pred_actions, _, _ = self.policy(states, None, None)
                pred_next_states = torch.zeros_like(states)
                for i, (state, action) in  enumerate( zip(states, pred_actions)):
                    self.envs.reset_model(state)
                    pred_next_state, _, _, _ = self.envs.step(action.detach())
                    pred_next_states[i] = pred_next_state
            
                action_loss = autils.compute_ac_loss(
                    pred_actions,
                    true_actions.view(-1, self.action_dim),
                    self.policy.action_space,
                )
                action_losses.append(action_loss.item())
                
                pred_loss, expert_loss = self.get_density(states, pred_next_states,
                pred_actions, true_next_states, true_actions)
                pred_density, expert_density = math.exp(-pred_loss), math.exp(-expert_loss)
                diff_loss = pred_loss - expert_loss
                
                diff_losses.append(diff_loss.item())

            return np.mean(action_losses + diff_losses)

    def update(self, storage):
        top_log_vals = super().update(storage) #actor_opt_lr
        log_vals = self._bc_step(True) #_pr_action_loss
        log_vals.update(top_log_vals) #_pr_action_loss
        return log_vals

    def get_storage_buffer(self, policy, envs, args):
        return BaseStorage()

    def get_add_args(self, parser):
        if not self.set_arg_defs:
            # This is set when BC is used at the same time as another optimizer
            # that also has a learning rate.
            self.set_arg_prefix("bc")

        super().get_add_args(parser)
        #########################################
        # Overrides
        if self.set_arg_defs:
            parser.add_argument("--num-processes", type=int, default=1)
            parser.add_argument("--num-steps", type=int, default=0)
            ADJUSTED_INTERVAL = 200
            parser.add_argument("--log-interval", type=int, default=ADJUSTED_INTERVAL)
            parser.add_argument(
                "--save-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
            parser.add_argument(
                "--eval-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
        parser.add_argument("--no-wb", default=False, action="store_true")

        #########################################
        # New args
        parser.add_argument("--bc-num-epochs", type=int, default=1)
        parser.add_argument("--bc-state-norm", type=str2bool, default=False)
        parser.add_argument("--bc-noise", type=float, default=None)
