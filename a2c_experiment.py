import useful_neural_nets
from A2C import A2cConductor, A2cConfig
import tensorflow as tf
import gym
import tensorflow.contrib.layers as layers

from utils import LinearSchedule
from run_dqn_atari import get_env
import numpy as np


class SimpleConfig(A2cConfig):
    def get_policy_function(self, features):
        return useful_neural_nets.policy_mlp(features, self.num_actions, hidden_layer_width=4)

    def get_value_function(self, features):
        return useful_neural_nets.value_function_mlp(features, 4)

    def get_conv_function(self, stacked_input_data):
        return layers.flatten(stacked_input_data)


def cartpole_test():
    def env_factory(nvm=False):
        return gym.make('CartPole-v0')

    session = tf.Session()
    config = SimpleConfig(env_factory, session, 150000)

    conductor = A2cConductor(config)

    model = conductor.run()

    for i in range(10):
        conductor.enjoy(model)

cartpole_test()

# def mountain_car_test():
#     def env_factory(nvm=False):
#         return gym.make('MountainCar-v0')
#
#     session = tf.Session()
#     config = SimpleConfig(env_factory, session, 500000)
#
#     conductor = A2cConductor(config)
#
#     model = conductor.run()
#
#     for i in range(10):
#         conductor.enjoy(model)
#
#
# mountain_car_test()



