from A3C import A3cConductor
import tensorflow as tf
import gym
import tensorflow.contrib.layers as layers


def env_factory():
    return gym.make('CartPole-v0')


def cartpole_conv_function(img_in, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("convnet", reuse=reuse):
            return layers.fully_connected(layers.flatten(img_in), num_outputs=32, activation_fn=tf.nn.relu)


session = tf.Session()
conductor = A3cConductor(env_factory, session)
conductor.config.conv_function = cartpole_conv_function
conductor.run()

