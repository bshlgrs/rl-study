import gym
import ddpg
import tensorflow as tf
import numpy as np
import utils

from run_dqn_atari import get_env


def train_pong():
    session = tf.Session()
    agent = ddpg.DdpgAgent(env=get_env(3, monitor=True), session=session, input_data_type=np.uint8)
    agent.learning_starts = 50000
    agent.num_timesteps = int(2e6)
    agent.exploration = utils.LinearSchedule(int(2e6), 0.1)
    agent.train()


# train_pong()


def train_cartpole():
    session = tf.Session()
    agent = ddpg.DdpgAgent(env=gym.make('CartPole-v0'), session=session, input_data_type=np.float32)
    agent.num_timesteps = 200000
    agent.learning_starts = 1000
    agent.exploration = utils.LinearSchedule(int(2e5), 0.1)
    agent.train()


train_cartpole()
