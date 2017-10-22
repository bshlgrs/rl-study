from collections import deque

import numpy as np


class EnvironmentWrapper:
    def __init__(self, n, env_factory, agent, config, model, render=False):
        super().__init__()
        self.n = n
        self.env = env_factory(n == 0)
        self.stop_signal = False
        self.agent = agent  # type: A2cAgent
        self.config = config  # type: A2cConfig
        self.queue = deque(maxlen=config.frame_history_len + 1)
        self.model = model  # type: A2cModel
        self.episode_reward = 0
        self.zero_state = np.zeros(self.config.state_shape)
        self.render = render
        self.done = False

        self.reset()

    def run_batch(self, steps):
        for _ in range(steps):
            self.step()

    def extract_experience(self):
        return self.agent.report_experience()

    def current_obs(self):
        return np.stack(list(self.queue)[1:])

    def prev_obs(self):
        return np.stack(list(self.queue)[:-1])

    def reset(self):
        for i in range(self.config.frame_history_len + 1):
            self.queue.append(self.zero_state)

        self.queue.append(self.env.reset())
        self.model.episodes_log.append(self.episode_reward)
        self.episode_reward = 0
        self.done = False

    def step(self):
        if self.done:
            self.reset()

        a = self.agent.act(self.current_obs())
        s_, r, done, _ = self.env.step(a)
        self.episode_reward += r
        self.done = done

        if self.render:
            self.env.render()

        if not done:
            self.queue.append(s_)

        self.agent.train(self.prev_obs(), a, r, self.current_obs(), done)