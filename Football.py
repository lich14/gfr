import numpy as np
import torch

from gfootball.env import football_action_set
from gym import Wrapper, spaces
from collections import deque


def get_feature_single(obs):
    ball_position = obs['ball']  # `ball` - [x, y, z] position of the ball.
    ball_position[1] = ball_position[1] / 0.42
    ball_position[2] = ball_position[2] / 10 - 1

    ball_direction = obs['ball_direction']  # `ball_direction` - [x, y, z] ball movement vector.

    ball_direction[0] = ball_direction[0] * 10
    ball_direction[1] = ball_direction[1] * 10
    ball_direction[2] = ball_direction[2] / 2

    ball_rotation = obs['ball_rotation']  # `ball_rotation` - [x, y, z] rotation angles in radians

    ball_rotation_sin = np.sin(ball_rotation)
    ball_rotation_cos = np.cos(ball_rotation)

    ball_owner = obs['ball_owned_player']  # `ball_owned_player` - {0..N-1} integer denoting index of the player owning the ball.

    current_position = obs['left_team'][1]  # left_team` - N-elements vector with [x, y] positions of players.

    current_position[1] = current_position[1] / 0.42

    current_direction = obs['left_team_direction'][1]  # `left_team_direction` - N-elements vector with [x, y] movement vectors of players.
    current_direction[0] = current_direction[0] * 50
    current_direction[1] = current_direction[1] * 50

    current_tired = obs['left_team_tired_factor'][
        1] * 2 - 1  # `left_team_tired_factor` - N-elements vector of floats in the range {0..1}. 0 means player is not tired at all.
    steps_left = obs['steps_left'] / 200 - 1  # `steps_left` - how many steps are left till the end of the match.

    return list_combine([
        ball_position, ball_direction, ball_rotation_sin, ball_rotation_cos,
        np.array([ball_owner]), current_position, current_direction,
        np.array([current_tired]),
        np.array([steps_left])
    ])


def list_combine(L):
    ref = L[0].tolist()
    for i in L[1:]:
        ref.extend(i.tolist())

    return ref


class Single_Wrapper(Wrapper):

    def __init__(self, env_, obs_stack=4):
        super(Single_Wrapper, self).__init__(env_)

        self.actions = football_action_set.action_set_v1
        self.obs_stack = obs_stack
        self.obs_dim = 19 * obs_stack
        self.action_dim = len(football_action_set.action_set_v1)
        self.distance_reward = True
        self.max_episode_length = 400
        self.distance_reward_discount_factor = 0.2
        self.loss_ball_reward = -0.05
        self.accumulate_reward_on_score = True

    def step(self, action_index):
        _, final_reward, done, info = self.env.step(self.actions[action_index])
        obs_original = self.env.unwrapped.observation()[0]
        obs = get_feature_single(obs_original)
        self.obs_buffer.append(obs)

        dense_reward = 0

        if self.accumulate_reward_on_score and final_reward == 1:
            dense_reward += obs_original['steps_left'] / self.max_episode_length

        if obs_original['ball_owned_player'] != 1:
            dense_reward += self.loss_ball_reward

        # o['ball'][0] is X, in the range [-1, 1]. o['ball'][1] is Y, in the range [-0.42, 0.42]
        # (2*2+0.42*0.42)**0.5 = 2.0436242316042352
        # the closer d to zero means the closer it is to the (enemy or right???) team's gate
        d = ((obs_original['ball'][0] - 1)**2 + obs_original['ball'][1]**2)**0.5
        # we divide by self.episode_limit since this reward is accumulative, we don't want the accumulative
        # reward to be too large after all.
        dense_reward += (0.5 - d / 2.0436242316042352) * self.distance_reward_discount_factor

        return np.hstack(self.obs_buffer), dense_reward + final_reward, done, info, final_reward

    def reset(self):
        self.env.reset()
        obs_original = self.env.unwrapped.observation()[0]
        obs = get_feature_single(obs_original)
        self.obs_buffer = deque([obs] * self.obs_stack, maxlen=self.obs_stack)

        return np.hstack(self.obs_buffer)
