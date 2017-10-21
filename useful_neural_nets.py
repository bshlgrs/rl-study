import tensorflow as tf
from tensorflow.contrib import layers as layers

import nice_helpers
import utils
import visualize_conv_kernels


def policy_mlp(input_features, num_actions, hidden_layer_width=512):
    policy_fc1 = layers.fully_connected(input_features, num_outputs=hidden_layer_width, activation_fn=tf.nn.relu)
    policy_fc2 = layers.fully_connected(policy_fc1, num_outputs=num_actions, activation_fn=None)
    policy_out = tf.nn.softmax(policy_fc2)

    nice_helpers.policy_argmax_summary(tf.argmax(policy_out, axis=1), num_actions)
    nice_helpers.policy_summaries(policy_out)

    return policy_out


def value_function_mlp(input_features, hidden_layer_width=512):
    value_fc1 = layers.fully_connected(input_features, num_outputs=hidden_layer_width, activation_fn=tf.nn.relu,
                                       scope='value_fc1')
    value_out = tf.reduce_sum(layers.fully_connected(value_fc1, num_outputs=1, activation_fn=None, scope='value_out'), axis=1,
                              name='value_out')

    utils.variable_summaries(value_out, 'value')

    with tf.variable_scope('value_fc1', reuse=True):
        utils.variable_summaries(tf.get_variable('weights'), 'value_fc1/weights', min_and_max=False)
        utils.variable_summaries(tf.get_variable('biases'), 'value_fc1/biases', min_and_max=False)

    with tf.variable_scope('value_out', reuse=True):
        utils.variable_summaries(tf.get_variable('weights'), 'value_out/weights', min_and_max=False)
        utils.variable_summaries(tf.get_variable('biases'), 'value_out/biases', min_and_max=False)
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

            # The following would be cool, but I haven't made it work yet.
                # The problem is that it doesn't know how to take into account the stacked frames that the conv filters
                # take in.
            # with tf.variable_scope('convnet', reuse=True):
            #     conv1_weights = tf.get_variable('Conv/weights')
            #     grid = visualize_conv_kernels.put_kernels_on_grid(conv1_weights)
            #     tf.summary.image('conv1/kernels', grid, max_outputs=32)

            out = layers.flatten(conv3)
            utils.variable_summaries(out, 'atari conv output', False)
            return out

