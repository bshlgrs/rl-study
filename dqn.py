import random

import tensorflow as tf
import numpy as np
import gym

from utils import weight_variable, bias_variable


## ALSO TODO: I think I'm supposed to consider errors in my weights from old q-values as well.

def make_graph():
    # To get predictions, call this with both values of a
    current_history = tf.placeholder([None, 84, 84, 4]) # is it easier to have this be a single vector here?
    a = tf.placeholder([None, 6])
    q_ = tf.placeholder([None])

    W_conv1 = weight_variable([8, 8, 1, 16]) # TODO: perhaps that 1 should be a 4
    b_conv1 = bias_variable([16])

    # TODO: figure out how to do this properly. current_history is probably shaped wrong.
    h1 = tf.nn.relu(tf.nn.conv2d(current_history, W_conv1, [1, 8, 8, 1], padding='SAME') + b_conv1)
    # The output there has 20*20 results for every filter, and 32 filters, and 4 frames
    # FUCK ME right

    # I have 32 4x4 filters
    W_conv2 = weight_variable([4, 4, 1, 32]) # TODO: perhaps that 1 should be a 4?
    b_conv2 = bias_variable([32])

    # I convolve them with SOME COMBINATION? of the outputs of the first layer of conv nets
    # This next line is totally wrong
    h2 = tf.nn.relu(tf.nn.conv2d(h1, W_conv2, [1, 2, 2, 1], padding='SAME') + b_conv2)

    # TODO: figure out how to make this map from all the different times in the history.
    W_fc3 = weight_variable([32, 4, 256]) # 32 filters on the

    loss = tf.reduce_mean(tf.nn.l2_loss(q - q_))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return [current_history, a, q, q_, train_step]


def make_one_hot(vector, num_classes):
    assert len(vector.shape) == 1
    size = vector.shape[0]

    return (np.tile(np.arange(num_classes), (size, 1)) ==
            np.broadcast_to(vector, (num_classes, size)).T).astype(np.float32)


def transform_frame(frame):
    # This is my guess as to a good place to crop the screen to; I could be wrong
    frame = frame[32:200]
    frame = frame[::2, ::2, 0]

    return frame  # shaped as an 84 x 84 region


def train():
    env = gym.make('Pong-v0')
    num_episodes = 1000

    # this is a list of tuples of type (four transformed frames, action, r, next transformed frame, done)
    replay_memory = []

    epsilon = 0.05
    num_actions = 6
    minibatch_size = 32
    gamma = 0.9

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        current_history, a, q, q_, train_step = make_graph()

        for i in range(num_episodes):
            sequence = []

            observation = env.reset()

            transformed_frame = transform_frame(observation)
            sequence.append(transformed_frame)
            done = False

            while not done:
                assert current_history.shape == (84, 84, 4)

                if random.random() < epsilon or len(sequence) < 4:
                    action = env.action_space.sample()
                else:
                    current_history_tiled = np.tile(current_history, (num_actions, 1, 1, 1))
                    assert current_history_tiled.shape == (6, 84, 84, 4)
                    trial_actions = np.eye(6)

                    action = sess.run([q], feed_dict={current_history: current_history_tiled, a: trial_actions})\
                                 .argmax()

                observation, reward, done, info = env.step(action)
                transformed_frame = transform_frame(observation)
                sequence.append(transformed_frame)

                if len(sequence) == 4:
                    current_history = np.stack(sequence[-4:], axis=-1)

                if len(sequence) > 4:
                    old_current_history = current_history
                    current_history = np.stack(sequence[-4:], axis=-1)

                    replay_memory.append((old_current_history, action, reward, current_history, done))

                # TRAINING.
                minibatch = random.sample(replay_memory, minibatch_size)

                # This is like [history0, history0, history1, history1, history2, history2 ...]
                minibatch_current_histories = np.array([
                    x[0]
                    for x in minibatch
                    for _ in range(num_actions)
                ])

                all_actions = np.concatenate(tuple([np.eye(num_actions) for _ in range(minibatch_size)]), axis=0)

                used_q_values = sess.run(q, feed_dict={
                                            current_history: minibatch_current_histories,
                                            a: all_actions})

                assert(used_q_values.shape == num_actions * minibatch_size)

                value_of_best_next_action = used_q_values.reshape((minibatch_size, num_actions)).max(axis=1)

                one_hot_actions = make_one_hot(np.array([x[1] for x in minibatch]), num_actions)
                rewards = np.array([x[2] for x in minibatch])
                done_array = np.array([x[4] for x in minibatch])
                target_q_values = rewards + gamma * value_of_best_next_action * (1 - done_array)

                minibatch_old_histories = [x[3] for x in minibatch]

                # now train
                sess.run(train_step, feed_dict={
                    current_history: minibatch_old_histories,
                    a: one_hot_actions,
                    q_: target_q_values
                })
