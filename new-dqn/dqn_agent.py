import itertools
import random
from collections import namedtuple, deque
from typing import List
import gym
import numpy as np
from types import *
from brain import Brain


class QLearningAgent:
    def __init__(self, num_actions, env):
        self.num_actions = num_actions
        self.env = env
        self.memory = Memory()
        self.brain = Brain()

    def get_action(self, obs: Observation, t: int) -> int:
        if random.random() < self.get_epsilon(t):
            return self.env.action_space.sample()
        else:
            return self.brain.predict(obs).argmax()

    @staticmethod
    def get_epsilon(t):
        if t < 1000000:
            return 1 - 0.9 * t/1000000
        else:
            return 0.1

    def observe(self, sample: Sample):
        self.memory.add(sample)

    def train(self, t):
        batch = self.memory.sample(32)
        self.brain.train(batch, t)


class Memory:
    def __init__(self):
        self.samples = deque(maxlen=1000000)

    def add(self, sample: Sample):
        self.samples.append(sample)

    def sample(self, n: int) -> List[Sample]:

        return [
            self.samples[random.randint(0, len(self.samples) - 1)] for _ in range(n)
        ]


def train(env: gym.Env, agent: QLearningAgent, max_timesteps: int):
    observation = env.reset()
    start_t = 0

    for t in range(max_timesteps):
        action = agent.get_action(observation, t)

        new_observation, reward, done, info = env.step(action)

        agent.observe(Sample(observation, action, reward, done, new_observation))
        agent.train(t)

        if done:
            new_observation = env.reset()
            print('time taken: %g' % (t - start_t))
            start_t = t

        observation = new_observation


def main():
    env = gym.make('CartPole-v0')

    agent = QLearningAgent(env.action_space.n)

    train(env, agent, 10000)


if __name__ == '__main__':
    main()