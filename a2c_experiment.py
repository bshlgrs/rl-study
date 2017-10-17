from A2C import A2cConductor
import tensorflow as tf
import gym
import tensorflow.contrib.layers as layers
from run_dqn_atari import get_env


def cartpole_test():
    def env_factory():
        return gym.make('CartPole-v0')


    def cartpole_conv_function(img_in, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # return layers.fully_connected(layers.flatten(img_in), num_outputs=32, activation_fn=tf.nn.relu)
            return layers.flatten(img_in)


    session = tf.Session()
    conductor = A2cConductor(env_factory, session)
    conductor.config.conv_function = cartpole_conv_function
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