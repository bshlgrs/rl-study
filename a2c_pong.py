import useful_neural_nets
from A2C import A2cConductor, A2cConfig
import tensorflow as tf
import gym
import tensorflow.contrib.layers as layers

from utils import LinearSchedule
from run_dqn_atari import get_env
import numpy as np


class PongConfig(A2cConfig):
    def get_policy_function(self, features):
        return useful_neural_nets.policy_mlp(features, self.num_actions, hidden_layer_width=512)

    def get_value_function(self, features):
        return useful_neural_nets.value_function_mlp(features, 512)

    def get_conv_function(self, stacked_input_data):
        return useful_neural_nets.atari_convnet(stacked_input_data, 'conv', reuse=None)


def cartpole_test():
    def env_factory(nvm=False):
        return gym.make('Pong-v0')

    session = tf.Session()
    config = PongConfig(env_factory, session, 2500000)

    conductor = A2cConductor(config)

    model = conductor.run()

    for i in range(10):
        conductor.enjoy(model)


cartpole_test()



