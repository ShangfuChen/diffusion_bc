
import pickle
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import tf_util
import gym
import load_policy
from rlf.exp_mgr.viz_utils import save_mp4
import torch
import os
import os.path as osp
from stable_baselines3 import PPO
tf.disable_eager_execution()

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

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--max-timesteps', type=int, default=300000)
    parser.add_argument('--num-rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--save-dir', type=str, default='../expert_datasets')
    parser.add_argument('--video-dir', type=str, default='./expert_vids')
    parser.add_argument('--noise-ratio', type=int, default=1)
    parser.add_argument('--record', type=bool, default=True)
    args = parser.parse_args()
    SAVE_DIR = args.save_dir
 
    with tf.Session():
        tf_util.initialize()
        
        print('loading and building expert policy')
        import gym
        env = gym.make(args.envname)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=int(2e5))
        model.save("./expert/dqn_lunar")
        print('loaded and built')

        max_steps = args.max_timesteps or env.spec.timestep_limit

        data = reset_data()
        episode = reset_data()
        returns = []
        if args.record:
            frames = [env.render("rgb_array")]
        for i in range(args.num_rollouts):
            print('iter', i)
            vec_env = model.get_env()
            obs = vec_env.reset()
            vec_env.seed(i)
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                next_obs, r, done, _ = vec_env.step(action)
                append_data(episode, obs, next_obs, action, done)
                if args.record and i==args.num_rollouts-1:
                    frames.append(vec_env.render("rgb_array"))
                obs = next_obs
                totalr += r
                steps += 1
                #if args.render:
                    #vec_env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            
            extend_data(data, episode)
            episode = reset_data()
            returns.append(totalr)
                    
            if args.record and i==args.num_rollouts-1:
            # frames = np.stack(frames)
                save_mp4(frames, args.video_dir, args.envname + "_%d" % (args.noise_ratio*100), fps=30, no_frame_drop=True)
                frames = []


        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        
        npify(data)
        
        save_name = args.envname.split("-")[0] + "_%d.pt" % (
        args.noise_ratio*100,
        )

        dones = data["done"]
        obs = data["obs"]
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

        '''
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        
        output_file_name = 'data/' + args.envname + '_' + str(args.num_rollouts) + '_data.pkl'
        with open(output_file_name, 'wb') as f:
            pickle.dump(expert_data, f)
        '''
if __name__ == '__main__':
    main()

