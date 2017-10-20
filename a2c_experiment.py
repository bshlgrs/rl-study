from A2C import A2cConductor
import tensorflow as tf
import gym
import tensorflow.contrib.layers as layers

from utils import LinearSchedule
from run_dqn_atari import get_env


def cartpole_test():
    def env_factory(nvm=False):
        return gym.make('CartPole-v0')

    def cartpole_conv_function(img_in, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # return layers.fully_connected(layers.flatten(img_in), num_outputs=64, activation_fn=tf.nn.relu)
            return layers.flatten(img_in)

    session = tf.Session()
    conductor = A2cConductor(env_factory, session)
    config = conductor.config

    config.conv_function = cartpole_conv_function
    config.exploration_schedule = LinearSchedule(150000, 0)
    config.num_steps = 150000
    config.input_data_type = tf.float32

    model = conductor.run()

    for i in range(10):
        conductor.enjoy(model)


def pong_test():
    def env_factory(monitor=False):
        return get_env(3, monitor=monitor)

    session = tf.Session()
    conductor = A2cConductor(env_factory, session)
    model = conductor.run()

    for i in range(10):
        conductor.enjoy(model)


pong_test()