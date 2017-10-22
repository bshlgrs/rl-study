import random
from collections import namedtuple

import numpy as np

MemoryItem = namedtuple("MemoryItem", ["s", "a", "r", "s_", "done"])


class ReusableAgent:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.memory = []
        self.gamma = 0.

    def act(self, obs):
        if random.random() < self.config.exploration(self.model.total_t):
            return random.randrange(self.config.num_actions)
        else:
            return np.random.choice(self.config.num_actions, p=self.model.get_policy(obs))

    def train(self, s, a, r, s_, done):
        self.memory.append(MemoryItem(s, a, r, s_, done))

    def report_experience(self):
        # compute n_step rewards, then send them to the master
        states = []
        actions = []
        total_rewards = []

        total_r = self.model.predict_values(np.array([self.memory[-1].s_]))[0]

        for item in reversed(self.memory):
            total_r = total_r * self.config.gamma + item.r

            states.append(item.s)
            actions.append(item.a)
            total_rewards.append(total_r)

            if item.done:
                total_r = 0

        self.memory = []
        return states, actions, total_rewards