"""
Main Loop of simulation.
Here a problem and a solution are defined.
"""

import gym
#from pyglet.window import key
import numpy as np

from tools import *
from base_solution import *

import os
import os.path as osp
import torch
from rlf.exp_mgr.viz_utils import save_mp4


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--envname', type=str, default='CarRacing-v0')
parser.add_argument('--render', action='store_true', default=True)
parser.add_argument('--max-timesteps', type=int, default=300000)
parser.add_argument('--n-episodes', type=int, default=30,
                    help='Number of expert roll outs')
parser.add_argument('--save-dir', type=str, default='../../expert_datasets')
parser.add_argument('--video-dir', type=str, default='../expert_vids')
parser.add_argument('--noise-ratio', type=int, default=1)
parser.add_argument('--record', type=bool, default=True)
args = parser.parse_args()
SAVE_DIR = args.save_dir

# Parameters
problem = args.envname

gym.logger.set_level(40)
all_episode_reward = []

# Initialize simulation
env = gym.make(problem)
env.reset()

# Define custom standard deviation for noise
# We can improve stability of solution, by noise parameters
noise_mean = np.array([0.0, -0.83], dtype=np.float32)
noise_std = np.array([0.0, 4 * 0.02], dtype=np.float32)
solution = BaseSolution(env.action_space, model_outputs=2, noise_mean=noise_mean, noise_std=noise_std)
solution.load_solution('models/best_solution/')

def reset_data():
    return {
        "obs": [],
        "next_obs": [],
        "actions": [],
        "done": [],
    }


def append_data(data, s, ns, a, done):
    data["obs"].append(s)
    data["next_obs"].append(ns)
    data["actions"].append(a)
    data["done"].append(done)


def extend_data(data, episode):
    data["obs"].extend(episode["obs"])
    data["next_obs"].extend(episode["next_obs"])
    data["actions"].extend(episode["actions"])
    data["done"].extend(episode["done"])
    
def npify(data):
    for k in data:
        if k == "dones":
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

data = reset_data()
episode = reset_data()
returns = []
if args.record:
    frames = [env.render("rgb_array")]
    
# Loop of episodes
for ie in range(args.n_episodes):
    env.seed(ie)
    state = env.reset()
    solution.reset()
    done = False
    episode_reward = 0
    no_reward_counter = 0

    # One-step-loop
    while not done:
        #env.render()

        action, train_action = solution.get_action(state, add_noise=True)
        # This will make steering much easier
        action /= 4
        new_state, reward, done, info = env.step(action)
        append_data(episode, state, new_state, action, done)
        if args.record and ie==args.n_episodes-1:
            frames.append(env.render("rgb_array"))

        state = new_state
        episode_reward += reward
         
        if reward < 0:
            no_reward_counter += 1
            if no_reward_counter > 200:
                break
        else:
            no_reward_counter = 0
    
    extend_data(data, episode)
    episode = reset_data()
    returns.append(episode_reward)
        
    if args.record and ie==args.n_episodes-1:
    # frames = np.stack(frames)
        save_mp4(frames, args.video_dir, args.envname + "_%d" % (args.noise_ratio*100), fps=30, no_frame_drop=True)
        frames = []

    all_episode_reward.append(episode_reward)
    average_result = np.array(all_episode_reward[-100:]).mean()
    print('Last result:', episode_reward, 'Average results:', average_result)

npify(data)

save_name = args.envname.split("-")[0] + "_%d.pt" % (
args.noise_ratio*100,
)

dones = data["done"]
obs = data["obs"]
print("")
print("obs.shape:", obs.shape)
print("")
next_obs = data["next_obs"]
actions = data["actions"]

torch.save(
    {
        "done": torch.FloatTensor(dones),
        "obs": torch.tensor(obs),
        "next_obs": torch.tensor(next_obs),
        "actions": torch.tensor(actions),
    },
    osp.join(SAVE_DIR, save_name),
)

env.close()

