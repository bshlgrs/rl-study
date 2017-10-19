import gym
import ddpg
import tensorflow as tf

from run_dqn_atari import get_env


def train_pong():
    session = tf.Session()
    agent = ddpg.DdpgAgent(env=get_env(3, monitor=True), session=session)
    agent.train()


train_pong()




