import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from rlf.algos.on_policy.on_policy_base import OnPolicy
from rlf.algos.il.density.ddpm import MLPDiffusion

# sample at any given time t, and calculate sampling loss
def diffusion_loss_fn(self, model, x_0_pred, x_0_expert, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0_pred.shape[0]
    
    # generate eandom t for a batch data
    t = torch.randint(0,n_steps,size=(batch_size//2,)).to(self.args.device)
    t = torch.cat([t,n_steps-1-t],dim=0) #[batch_size, 1]
    t = t.unsqueeze(-1)
    
    # coefficient of x0
    a = alphas_bar_sqrt[t]
    
    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]
    
    # generate random noise eps
    e = torch.randn_like(x_0_pred)
    e2 = torch.randn_like(x_0_expert)
    
    # model input
    x = x_0_pred*a + e*aml
    x2 = x_0_expert*a + e2*aml
    
    # get predicted randome noise at time t
    output = model(x, t.squeeze(-1))
    output2 = model(x2, t.squeeze(-1))
    
    # calculate the loss between actual noise and predicted noise
    loss = (e - output).square().mean()
    loss2 = (e2 - output2).square().mean()
    
    return loss, loss2

def get_density(self, states, pred_action, expert_action):
    num_steps = 100
    dim = states.size()[1] + pred_action.size()[1]
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
    
    pred = torch.cat((states, pred_action), 1)
    expert = torch.cat((states, expert_action), 1)
    pred_loss, expert_loss = self.diffusion_loss_fn(model, pred, expert, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
    
    return pred_loss, expert_loss

class PPO(OnPolicy):
    def __init__(self, algo_name='else'):
        super().__init__()
        self.algo_name = algo_name

    def update(self, rollouts):
        
        self._compute_returns(rollouts)
        advantages = rollouts.compute_advantages()

        use_clipped_value_loss = True

        log_vals = defaultdict(lambda: 0)

        for e in range(self._arg('num_epochs')):
            data_generator = rollouts.get_generator(advantages,
                    self._arg('num_mini_batch'))

            for sample in data_generator:
                # Get all the data from our batch sample
                ac_eval = self.policy.evaluate_actions(sample['state'],
                        sample['other_state'],
                        sample['hxs'], sample['mask'],
                        sample['action'])

                ratio = torch.exp(ac_eval['log_prob'] - sample['prev_log_prob'])
                surr1 = ratio * sample['adv']
                surr2 = torch.clamp(ratio,
                        1.0 - self._arg('clip_param'),
                        1.0 + self._arg('clip_param')) * sample['adv']
                action_loss = -torch.min(surr1, surr2).mean(0)

                if use_clipped_value_loss:
                    value_pred_clipped = sample['value'] + (ac_eval['value'] - sample['value']).clamp(
                                    -self._arg('clip_param'),
                                    self._arg('clip_param'))
                    value_losses = (ac_eval['value'] - sample['return']).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - sample['return']).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (sample['return'] - ac_eval['value']).pow(2).mean()

                loss = (value_loss * self._arg('value_loss_coef') + action_loss -
                     ac_eval['ent'].mean() * self._arg('entropy_coef'))

                '''   
                if self.algo_name == 'GAIL':
                    dim = sample['state'].size()[1] + sample['action'].size()[1]
                    num_steps = 100
                    model = MLPDiffusion(num_steps, input_dim = dim).to(self.args.device)
                    pred_loss, expert_loss = get_density(sample['state'], pred_actions, true_actions)
                    diff_loss = pred_loss - expert_loss
                    loss = loss + diff_loss
                '''

                self._standard_step(loss)

                log_vals['value_loss'] += value_loss.sum().item()
                log_vals['action_loss'] += action_loss.sum().item()
                log_vals['dist_entropy'] += ac_eval['ent'].mean().item()

        num_updates = self._arg('num_epochs') * self._arg('num_mini_batch')
        for k in log_vals:
            log_vals[k] /= num_updates

        return log_vals

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(f"--{self.arg_prefix}clip-param",
            type=float,
            default=0.2,
            help='ppo clip parameter')

        parser.add_argument(f"--{self.arg_prefix}entropy-coef",
            type=float,
            default=0.01,
            help='entropy term coefficient (old default: 0.01)')

        parser.add_argument(f"--{self.arg_prefix}value-loss-coef",
            type=float,
            default=0.5,
            help='value loss coefficient')

