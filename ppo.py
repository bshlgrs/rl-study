import tensorflow as tf


def l_clip(policy, policy_old, advantage, one_hot_actions, epsilon):
    r_t = tf.reduce_sum(one_hot_actions * policy / policy_old, axis=1)

    unclipped_objective = r_t * advantage

    clipped_r_t = tf.where(r_t > 1 + epsilon,
                           1 + epsilon,
                           tf.where(
                               r_t < 1 - epsilon,
                               1 - epsilon,
                               r_t))

    clipped_objective = clipped_r_t * advantage

    min_objective = tf.minimum(unclipped_objective, clipped_objective)

    return tf.reduce_mean(min_objective, name="L_clip")

