import tensorflow as tf
import numpy as np
import utils


def obs_nodes(input_data_type, input_shape, name):
    if input_data_type == np.float32:
        obs_ph = tf.placeholder(tf.float32, [None] + list(input_shape), name=name)
        obs_float = obs_ph
    elif input_data_type == np.uint8:
        obs_ph = tf.placeholder(tf.uint8, [None] + list(input_shape), name=name)
        obs_float = tf.cast(obs_ph, tf.float32) / 255.0
    else:
        raise NotImplementedError

    return obs_ph, obs_float


def policy_argmax_summary(act_t_ph, num_actions):
    action_probabilities = tf.reduce_mean(tf.one_hot(act_t_ph, num_actions), axis=0)
    for i in range(num_actions):
        tf.summary.scalar('action_probs/action_%d_prob' % i, action_probabilities[i])


def policy_summaries(policy_out):
    policy_min = tf.reduce_min(policy_out, axis=1)
    policy_max = tf.reduce_max(policy_out, axis=1)
    utils.variable_summaries(policy_min, 'policy_min', min_and_max=False)
    utils.variable_summaries(policy_max, 'policy_max', min_and_max=False)


def mean_square_error(x, y):
    return tf.reduce_mean(tf.square(x - y))


def add_misc_summary(data, t, writer):
    summary = tf.Summary()
    for key, value in data.items():
        summary.value.add(tag='misc/' + key, simple_value=value)
    writer.add_summary(summary, t)
