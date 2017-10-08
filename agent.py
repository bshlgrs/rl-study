import sys
from datetime import datetime

import gym.spaces
import itertools
import numpy as np
import random
from dqn_utils import *
import models


# replay_buffer_size = 1000000,
# batch_size = 128,
# gamma = 0.99,
# learning_starts = 50000,
# learning_freq = 16,
# target_update_freq = 10000,
# grad_norm_clipping = 10):
# frame_history_len = 4

class DQNAgent:
    def __init__(self, env, session, batch_size=512):
        self.env = env
        self.replay_buffer_size = int(1e6)
        self.gamma = 0.99
        self.learning_starts = 50000

        self.exploration = LinearSchedule(1000000, 0.1)
        self.stopping_criterion = None
        self.model = models.Model(session, env, batch_size=batch_size)
        self.target_update_freq = 10000

        self.learning_freq = 4 * batch_size / 32

    def learn(self, num_timesteps):
        assert type(self.env.observation_space) == gym.spaces.Box
        assert type(self.env.action_space) == gym.spaces.Discrete

        replay_buffer = ReplayBuffer(self.replay_buffer_size, self.model.frame_history_len)

        mean_episode_reward = -float('nan')
        log_rate = 10000

        last_obs = self.env.reset()
        done = False
        q_value_log = MeanLogger()

        for t in range(num_timesteps):
            if done:
                last_obs = self.env.reset()

            idx = replay_buffer.store_frame(last_obs)

            if random.random() < self.exploration.value(t) or not self.model.model_initialized:
                action = self.env.action_space.sample()
            else:
                best_action = self.model.choose_best_action(replay_buffer.encode_recent_observation())
                action = best_action.action_idx
                q_value_log.log(best_action.q_value)

            obs, reward, done, info = self.env.step(action)
            replay_buffer.store_effect(idx, action, reward, done)
            last_obs = obs

            if (t > self.learning_starts and
                    t % self.learning_freq == 0 and
                    replay_buffer.can_sample(self.model.batch_size)):
                self.model.train(replay_buffer.sample(self.model.batch_size), t)

            if t % self.target_update_freq == 0:
                self.model.update_target_network()

            episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if t % log_rate == 0:
                self.report(locals())

        self.report(locals())

    @memoized
    def exploration_schedule(self):
        return PiecewiseSchedule(
            [
                (0, 1.0),
                (1e6, 0.1),
                # should be num_iterations
                (4e6 / 2, 0.01),
            ], outside_value=0.01
        )

    def report(self, variables):
        print("%d,%f,%d,%s,%f,%f" % (
            variables['t'],
            variables['mean_episode_reward'],
            len(variables['episode_rewards']),
            datetime.now(),
            variables['q_value_log'].pop(),
            self.model.current_learning_rate(variables['t'])
        ))

        sys.stdout.flush()
