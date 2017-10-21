import tensorflow as tf
from tensorflow.contrib import layers as layers

import nice_helpers
import utils


def policy_mlp(input_features, num_actions, hidden_layer_width=512):
    policy_fc1 = layers.fully_connected(input_features, num_outputs=hidden_layer_width, activation_fn=tf.nn.relu)
    policy_fc2 = layers.fully_connected(policy_fc1, num_outputs=num_actions, activation_fn=None)
    policy_out = tf.nn.softmax(policy_fc2)

    nice_helpers.policy_argmax_summary(tf.argmax(policy_out, axis=1), num_actions)
    nice_helpers.policy_summaries(policy_out)

    return policy_out


def value_function_mlp(input_features, hidden_layer_width=512):
    value_fc1 = layers.fully_connected(input_features, num_outputs=hidden_layer_width, activation_fn=tf.nn.relu)
    value_out = tf.reduce_sum(layers.fully_connected(value_fc1, num_outputs=1, activation_fn=None), axis=1,
                              name='value_out')

    utils.variable_summaries(value_out, 'value')
    return value_out


def policy_linear(input_features, num_actions):
    policy_logits = layers.fully_connected(input_features, num_outputs=num_actions, activation_fn=None)
    policy_out = tf.nn.softmax(policy_logits)

    nice_helpers.policy_argmax_summary(tf.argmax(policy_out, axis=1), num_actions)
    nice_helpers.policy_summaries(policy_out)

    return policy_out


def atari_convnet(img_in, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("convnet", reuse=reuse):
            conv1 = layers.convolution2d(img_in, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            conv2 = layers.convolution2d(conv1, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            conv3 = layers.convolution2d(conv2, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
    return layers.flatten(conv3)
