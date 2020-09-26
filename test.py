import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import gfootball.env as football_env

env = football_env.create_environment(
    env_name='academy_3_vs_1_with_keeper',
    render=True,
    stacked=False,
    representation='simple115v2',
    number_of_left_players_agent_controls=3,
)
env.reset()
env.render()

for i in range(100):
    print(i)
    env.step([1, 1, 1])
    env.render()
