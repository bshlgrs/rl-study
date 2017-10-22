import random

import datetime
import tensorflow as tf

import nice_helpers
import replay_buffer
import utils
import tensorflow.contrib.layers as layers
import dqn_models
import numpy as np
import useful_neural_nets

class DdpgAgent:
    def __init__(self,
                 env=None,
                 session=None,
                 buffer_size=int(1e6),
                 frame_history_len=1,
                 input_data_type=None):
        assert env is not None
        self.env = env
        self.session = session
        self.frame_history_len = frame_history_len
        self.buffer_size = buffer_size
        self.input_shape = env.observation_space.shape
        self.gamma = 0.99
        self.num_timesteps = None
        self.exploration = None # utils.LinearSchedule(int(1e5), 0.1)
        self.learning_starts = None
        self.batch_size = 64
        self.learning_freq = 4 * self.batch_size / 32
        self.input_data_type = input_data_type
        self.learning_rate = 0.001
        assert input_data_type == np.uint8 or input_data_type == np.float32

    def train(self):
        model = DdpgModel(self, self.session)

        buffer = replay_buffer.ReplayBuffer(self.buffer_size, self.frame_history_len, dtype=np.float32)

        last_obs = self.env.reset()
        done = False
        episode_rewards = []
        this_episode_reward = 0
        rewards_reported_on = 0

        for t in range(self.num_timesteps):
            if t % 1000 == 0 and t > 0:
                new_rewards = episode_rewards[rewards_reported_on:]
                if len(new_rewards) > 0:
                    mean_reward = sum(new_rewards) / len(new_rewards)
                    model.add_misc_summary({
                        'average_episode_reward': mean_reward,
                        'num_episodes': len(episode_rewards),
                        'epsilon': self.exploration.value(t)
                    }, t)

            if done:
                episode_rewards.append(this_episode_reward)
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
        # type: (DdpgAgent, tf.Session) -> None
        self.agent = agent

        gamma = agent.gamma
        num_actions = agent.env.action_space.n

        if agent.input_data_type == np.float32:
            obs = tf.placeholder(tf.float32, [None] + list(agent.input_shape), name='obs')
            obs_tp1 = tf.placeholder(tf.float32, [None] + list(agent.input_shape), name='obs_tp1')
            obs_float = obs
            obs_tp1_float = obs_tp1
        else:
            obs = tf.placeholder(tf.uint8, [None] + list(agent.input_shape), name='obs')
            obs_tp1 = tf.placeholder(tf.uint8, [None] + list(agent.input_shape), name='obs_tp1')
            obs_float = tf.cast(obs, tf.float32) / 255.0
            obs_tp1_float = tf.cast(obs_tp1, tf.float32) / 255.0

        act = tf.placeholder(tf.uint8, [None], name='act')
        rew = tf.placeholder(tf.float32, [None], name='rew')
        done = tf.placeholder(tf.float32, [None], name='done_mask')
        nice_helpers.policy_argmax_summary(act, num_actions)

        tau = tf.placeholder(tf.float32, [], name="tau")
        learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")

        actor, critic = DdpgModel.get_networks(obs_float, agent.env.action_space.n, 'model', False)
        actor_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/actor')
        critic_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/critic')
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')

        actor_target, critic_target = DdpgModel.get_networks(obs_tp1_float, num_actions, 'target_model', False)
        update_target_fn = nice_helpers.soft_update_target_fn('model', 'target_model', tau)

        def policy(s):
            return session.run(actor, feed_dict={obs: s})

        self.policy = policy

        def choose_action(s):
            return np.argmax(policy(np.array([s]))[0])

        self.choose_action = choose_action

        y = rew + gamma * tf.reduce_sum(actor_target * critic_target, axis=1) * (1 - done)
        utils.variable_summaries(y, 'y')
        utils.scalar_summary('done_mean', tf.reduce_mean(done))

        with tf.variable_scope('loss'):
            one_hot_actions = tf.one_hot(act, num_actions)

            critic_loss = tf.reduce_mean(tf.square(y - tf.reduce_sum(critic * one_hot_actions, axis=1)),
                                         name='critic_loss')
            utils.scalar_summary('critic_loss', critic_loss)

            neg_log_policy = -tf.log(actor + 1e-10) * one_hot_actions
            policy_loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * tf.stop_gradient(critic), axis=1),
                                         name='policy_loss')
            utils.scalar_summary('policy_loss', policy_loss)

        learning_rate_batch_size_correction = agent.batch_size / 32
        critic_learning_rate = learning_rate * learning_rate_batch_size_correction
        critic_update = DdpgModel.make_optimizer_step(critic_func_vars, critic_loss, critic_learning_rate, 0.5)
        # this is what the cool kids do
        actor_learning_rate = learning_rate * learning_rate_batch_size_correction / 10
        actor_update = DdpgModel.make_optimizer_step(actor_func_vars, policy_loss, actor_learning_rate, 0.5)


        merged_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/tmp/train/' + str(datetime.datetime.now()) + "/", session.graph)

        outside_locals = locals()

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

        def add_misc_summary(data, t):
            nice_helpers.add_misc_summary(data, t, train_writer)
        self.add_misc_summary = add_misc_summary

        session.run(tf.global_variables_initializer())

    def current_learning_rate(self, t):
        return self.agent.learning_rate

    @staticmethod
    def get_networks(obs_float, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # conv
            if obs_float.get_shape().ndims == 2:
                conv_flattened = layers.flatten(obs_float)
            else:
                conv_flattened = useful_neural_nets.atari_convnet(obs_float, 'conv', reuse)

            with tf.variable_scope('actor'):
                actor = useful_neural_nets.policy_mlp(conv_flattened, num_actions)
            with tf.variable_scope('critic'):
                critic = useful_neural_nets.advantage_function_mlp(conv_flattened, num_actions)

        return [actor, critic]

    @staticmethod
    def make_optimizer_step(func_vars, loss, learning_rate, gradient_clip):
        clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(loss, func_vars), gradient_clip)
        grads = list(zip(clipped_grads, func_vars))

        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-5)
        return optimizer.apply_gradients(grads)
