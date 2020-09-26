import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import gfootball.env as football_env

from SAC import SAC
from Football import Single_Wrapper
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--scenario', type=str, default='academy_empty_goal', help='environment')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate (default: 0.0003)')
parser.add_argument(
    '--alpha',
    type=float,
    default=1,
    metavar='G',
    help='Temperature parameter alpha determines the relative importance of the entropy\
                            term against the reward (default: 1)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G', help='Automaically adjust alpha (default: False)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N', help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N', help='Steps sampling random actions (default: 10000)')
parser.add_argument(
    '--target_update_interval', type=int, default=1, metavar='N', help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N', help='size of replay buffer (default: 10000000)')
parser.add_argument('--GPU', type=str, default="cuda:1", help='bool')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = football_env.create_environment(
    env_name=args.scenario,
    render=False,
    stacked=True,
    representation='simple115',
)
env.seed(args.seed)
env = Single_Wrapper(env)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.obs_dim, env.action_dim, args)

#TesnorboardX
writer = SummaryWriter(logdir='runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.scenario,
                                                         "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
test_step = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = np.random.randint(env.action_dim)  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _, final_reward = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.max_episode_length else float(not done)

        memory.push(state, action, reward, next_state, mask)  # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, final reward: {}".format(
        i_episode, total_numsteps, episode_steps, round(episode_reward, 2), final_reward))

    if total_numsteps - test_step >= 5000:
        avg_reward = 0.
        avg_final_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)
                next_state, reward, done, _, final_reward = env.step(action)
                episode_reward += reward
                state = next_state

            avg_reward += episode_reward
            avg_final_reward += final_reward
        avg_reward /= episodes
        avg_final_reward /= episodes

        writer.add_scalar('test/avg_reward', avg_reward, total_numsteps)
        writer.add_scalar('test/avg_final_reward', avg_final_reward, total_numsteps)

        print("----------------------------------------")
        print("Test Steps: {}, Avg. Reward: {}".format(total_numsteps, round(avg_reward, 2)))
        print("Test Steps: {}, Avg. Final Reward: {}".format(total_numsteps, round(avg_final_reward, 2)))
        print("----------------------------------------")
        test_step += 5000
        if test_step % 100000 == 0:
            agent.save_model(args.automatic_entropy_tuning, test_step)
