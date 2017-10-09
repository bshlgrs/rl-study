import sys
from datetime import datetime

import gym.spaces
import itertools
import numpy as np
import random
from dqn_utils import *
import models


class DQNAgent:
    def __init__(self, env, session, batch_size=32):
        self.env = env
        self.replay_buffer_size = int(1e6)
        self.gamma = 0.99
        self.learning_starts = 50000
        self.exploration = LinearSchedule(1000000, 0.1)
        self.stopping_criterion = None
        self.model = models.Model(session, env, batch_size=batch_size)
        self.target_update_freq = 10000
        self.log_rate = 10000
        self.learning_freq = 4 * batch_size / 32

    def learn(self, num_timesteps):
        assert type(self.env.observation_space) == gym.spaces.Box
        assert type(self.env.action_space) == gym.spaces.Discrete

        replay_buffer = ReplayBuffer(self.replay_buffer_size, self.model.frame_history_len)

        mean_episode_reward = -float('nan')

        last_obs = self.env.reset()
        done = False

        for t in range(num_timesteps):
            if done:
                last_obs = self.env.reset()

            idx = replay_buffer.store_frame(last_obs)

            if random.random() < self.exploration.value(t) or not self.model.model_initialized:
                action = self.env.action_space.sample()
            else:
                best_action = self.model.choose_best_action(replay_buffer.encode_recent_observation())
                action = best_action.action_idx

            obs, reward, done, info = self.env.step(action)
            replay_buffer.store_effect(idx, action, reward, done)
            last_obs = obs

            if t > self.learning_starts:
                if t % self.learning_freq == 0 and replay_buffer.can_sample(self.model.batch_size):
                    self.model.train(replay_buffer.sample(self.model.batch_size), t)

                if t % self.target_update_freq == 0:
                    self.model.update_target_network()

            episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])

            if t % self.log_rate == 0:
                info = {
                    'epsilon': self.exploration.value(t),
                    'mean_episode_reward': mean_episode_reward,
                    'num_episodes': len(episode_rewards),
                    'learning_rate': self.model.current_learning_rate(t)
                }

                self.model.log(info, t)
