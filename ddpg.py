import random

import tensorflow as tf
import replay_buffer
import utils
import tensorflow.contrib.layers as layers
import models
import numpy as np


class DdpgAgent:
    def __init__(self, env=None, session=None, buffer_size=int(1e6), frame_history_len=1):
        assert env is not None
        self.env = env
        self.session = session
        self.frame_history_len = frame_history_len
        self.buffer_size = buffer_size
        self.input_shape = env.observation_space.shape
        self.gamma = 0.99
        self.num_timesteps = 2000000
        self.exploration = utils.LinearSchedule(100000, 0.1)
        self.learning_starts = 40000
        self.batch_size = 512
        self.learning_freq = 4 * self.batch_size / 32

    def train(self):
        model = DdpgModel(self, self.session)

        buffer = replay_buffer.ReplayBuffer(self.buffer_size, self.frame_history_len)

        last_obs = self.env.reset()
        done = False
        episode_rewards = []
        this_episode_reward = 0

        for t in range(self.num_timesteps):
            if t % 10000 == 0:
                print("timestep", t)
            if done:
                episode_rewards.append(this_episode_reward)
                print('this episode reward', this_episode_reward)
                this_episode_reward = 0
                last_obs = self.env.reset()

            idx = buffer.store_frame(last_obs)

            if random.random() < self.exploration.value(t):
                action = self.env.action_space.sample()
            else:
                action = model.choose_action(buffer.encode_recent_observation())

            obs, reward, done, info = self.env.step(action)
            this_episode_reward += reward
            buffer.store_effect(idx, action, reward, done)
            last_obs = obs

            if t > self.learning_starts:
                if t % self.learning_freq == 0 and buffer.can_sample(self.batch_size):
                    s, a, r, s_, done_mask = buffer.sample(self.batch_size)

                    model.update(s, a, r, s_, done_mask, t)


class DdpgModel:
    def __init__(self, agent, session):
        # type: (DdpgAgent) -> None

        gamma = agent.gamma
        num_actions = agent.env.action_space.n

        obs = tf.placeholder(tf.uint8, [None] + list(agent.input_shape), name='obs')
        obs_tp1 = tf.placeholder(tf.uint8, [None] + list(agent.input_shape), name='obs_tp1')
        act = tf.placeholder(tf.uint8, [None], name='act')
        rew = tf.placeholder(tf.float32, [None], name='rew')
        done = tf.placeholder(tf.float32, [None], name='done_mask')
        obs_float = tf.cast(obs, tf.float32) / 255.0
        obs_tp1_float = tf.cast(obs_tp1, tf.float32) / 255.0

        tau = tf.placeholder(tf.float32, [], name="tau")
        learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")

        actor, critic = DdpgModel.get_networks(obs_float, agent.env.action_space.n, 'model', False)
        actor_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/actor')
        critic_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/critic')
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')

        actor_target, critic_target = DdpgModel.get_networks(obs_tp1_float, num_actions, 'target_model', False)

        utils.variable_summaries(actor, 'actor')
        utils.variable_summaries(rew, 'rew')
        utils.variable_summaries(critic, 'critic')
        action_probabilities = tf.reduce_mean(actor, axis=0)
        for i in range(num_actions):
            tf.summary.scalar('action_%d_prob'%i, action_probabilities[i])

        def policy(s):
            return session.run(actor, feed_dict={obs: s})

        self.policy = policy

        def choose_action(s):
            return np.random.choice(num_actions, p=policy(np.array([s]))[0])

        self.choose_action = choose_action

        y = rew + gamma * tf.reduce_sum(actor_target * critic_target, axis=1) * (1 - done)
        utils.variable_summaries(y, 'y')

        one_hot_actions = tf.one_hot(act, num_actions)
        critic_loss = tf.reduce_mean(tf.square(y - tf.reduce_sum(critic * one_hot_actions, axis=1)) * (1 - done),
                                     name='critic_loss')
        utils.scalar_summary('critic_loss')

        neg_log_policy = -tf.log(actor + 1e-10) * one_hot_actions
        policy_loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * tf.stop_gradient(critic), axis=1))
        utils.scalar_summary('policy_loss')

        critic_update = DdpgModel.make_optimizer_step(critic_func_vars, critic_loss, learning_rate, 0.5)
        # this is what the cool kids do
        actor_learning_rate = learning_rate / 10
        actor_update = DdpgModel.make_optimizer_step(actor_func_vars, policy_loss, actor_learning_rate, 0.5)

        target_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_model')

        update_target_fn = []
        for var, var_target in zip(sorted(model_vars, key=lambda v: v.name),
                                   sorted(target_net_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(tau * var + (1 - tau) * var_target))

        update_target_fn = tf.group(*update_target_fn, name='update_target_fn')

        merged_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/tmp/train/', session.graph)

        def update(s, a, r, s_, done_mask, t):
            summary_result, _, _, _ = session.run([merged_summaries, critic_update, actor_update, update_target_fn],
                        feed_dict={obs: s,
                                   act: a,
                                   rew: r,
                                   obs_tp1: s_,
                                   learning_rate: self.current_learning_rate(t),
                                   done: done_mask,
                                   tau: 0.001})
            train_writer.add_summary(summary_result, t)

        self.update = update

        session.run(tf.global_variables_initializer())

    def current_learning_rate(self, t):
        return 0.001

    @staticmethod
    def get_networks(obs_float, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # conv
            with tf.variable_scope('conv'):
                conv1 = layers.convolution2d(obs_float, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
                conv2 = layers.convolution2d(conv1, num_outputs=64, kernel_size=2, stride=2, activation_fn=tf.nn.relu)
                conv3 = layers.convolution2d(conv2, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
                conv_flattened = layers.flatten(conv3)

            with tf.variable_scope('actor'):
                actor1 = layers.fully_connected(conv_flattened, num_outputs=512, activation_fn=tf.nn.relu)
                actor2 = layers.fully_connected(actor1, num_outputs=num_actions, activation_fn=None)
                actor = tf.nn.softmax(actor2)

            with tf.variable_scope('critic'):
                critic1 = layers.fully_connected(conv_flattened, num_outputs=512, activation_fn=tf.nn.relu)
                critic = layers.fully_connected(critic1, num_outputs=num_actions, activation_fn=None)

        return [actor, critic]

    @staticmethod
    def make_optimizer_step(func_vars, loss, learning_rate, gradient_clip):
        clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(loss, func_vars), gradient_clip)
        grads = list(zip(clipped_grads, func_vars))

        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-5)
        return optimizer.apply_gradients(grads)

