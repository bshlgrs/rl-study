import random
from collections import namedtuple, deque

from datetime import datetime
from gym.spaces import Box
from tensorflow.contrib import layers
import tensorflow as tf

import nice_helpers
import utils
import numpy as np
from useful_neural_nets import atari_convnet

MemoryItem = namedtuple("MemoryItem", ["s", "a", "r", "s_", "done"])


class A2cEnvironmentWrapper:
    def __init__(self, n, env_factory, agent, config, model, render=False):
        super().__init__()
        self.n = n
        self.env = env_factory(n == 0)
        self.stop_signal = False
        self.agent = agent  # type: A2cAgent
        self.config = config  # type: A2cConfig
        self.queue = deque(maxlen=config.frame_history_len + 1)
        self.model = model  # type: A2cModel
        self.episode_reward = 0
        self.zero_state = np.zeros(self.config.state_shape)
        self.render = render
        self.done = False

        self.reset()

    def run_batch(self, steps):
        for _ in range(steps):
            self.step()

    def extract_experience(self):
        return self.agent.report_experience()

    def current_obs(self):
        return np.stack(list(self.queue)[1:])

    def prev_obs(self):
        return np.stack(list(self.queue)[:-1])

    def reset(self):
        for i in range(self.config.frame_history_len + 1):
            self.queue.append(self.zero_state)

        self.queue.append(self.env.reset())
        self.model.episodes_log.append(self.episode_reward)
        self.episode_reward = 0
        self.done = False

    def step(self):
        if self.done:
            self.reset()

        a = self.agent.act(self.current_obs())
        s_, r, done, _ = self.env.step(a)
        self.episode_reward += r
        self.done = done

        if self.render:
            self.env.render()

        if not done:
            self.queue.append(s_)

        self.agent.train(self.prev_obs(), a, r, self.current_obs(), done)


class A2cAgent:
    def __init__(self, config, model):
        self.config = config
        self.model = model  # type: A2cModel
        self.memory = []
        self.gamma = 0.

    def act(self, obs):
        if random.random() < self.config.exploration(self.model.total_t):
            return random.randrange(self.config.num_actions)
        else:
            return np.random.choice(self.config.num_actions, p=self.model.get_policy(obs))

    def train(self, s, a, r, s_, done):
        self.memory.append(MemoryItem(s, a, r, s_, done))

    def report_experience(self):
        # compute n_step rewards, then send them to the master
        states = []
        actions = []
        total_rewards = []

        total_r = self.model.predict_values(np.array([self.memory[-1].s_]))[0]

        for item in reversed(self.memory):
            total_r = total_r * self.config.gamma + item.r

            states.append(item.s)
            actions.append(item.a)
            total_rewards.append(total_r)

            if item.done:
                total_r = 0

        self.memory = []
        return states, actions, total_rewards


class A2cModel(object):
    def __init__(self, config, conductor):
        self.config = config
        self.conductor = conductor

        self.conv_function = self.config.conv_function
        self.session = self.config.session

        self.episodes_log = []

        self.total_t = 0
        num_actions = config.num_actions

        obs_float, obs_ph = nice_helpers.obs_nodes(config.input_data_type, config.input_shape, 'obs')
        act_t_ph = tf.placeholder(tf.int32, [None], name='act_t_ph')
        return_t_ph = tf.placeholder(tf.float32, [None], name='return_t_ph')
        learning_rate_ph = tf.placeholder(tf.float32, (), name="learning_rate_ph")

        nice_helpers.policy_argmax_summary(act_t_ph, num_actions)

        with tf.variable_scope('model'):

            conv_out = config.get_conv_function(obs_float)

            with tf.variable_scope('policy'):
                policy_out = config.get_policy_function(conv_out)

                with tf.variable_scope('entropy'):
                    entropy = tf.reduce_mean(-tf.log(policy_out + 1e-10) * policy_out) * num_actions
                tf.summary.scalar('entropy', entropy)

            with tf.variable_scope('value'):
                value_out = config.get_value_function(conv_out)

        with tf.variable_scope('losses'):
            neg_log_p_ac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_out, labels=act_t_ph)

            empirical_advantage = return_t_ph - value_out
            utils.variable_summaries(empirical_advantage, 'empirical_advantage')

            policy_loss = tf.reduce_mean(neg_log_p_ac * tf.stop_gradient(empirical_advantage))
            tf.summary.scalar('policy_loss', policy_loss)

            value_loss = nice_helpers.mean_square_error(tf.squeeze(value_out), return_t_ph)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('value_prediction_and_true_value_covariance', utils.covariance(value_out, return_t_ph))

            total_loss = tf.reduce_sum(policy_loss +
                                       config.value_loss_constant * value_loss -
                                       config.regularization_constant * entropy, name="total_loss")
            tf.summary.scalar('total_loss', total_loss)

        merged_summaries = tf.summary.merge_all()

        params = utils.find_trainable_variables('model')
        clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, params), 0.5)
        grads = list(zip(clipped_grads, params))

        optimizer = tf.train.RMSPropOptimizer(learning_rate_ph, decay=.99, epsilon=1e-5)
        _train = optimizer.apply_gradients(grads)

        def get_policy(observation):
            return self.session.run(policy_out, feed_dict={obs_ph: np.array([observation])})[0]

        self.get_policy = get_policy

        def train(s, a, r):
            s = np.stack(s)
            a = np.stack(a)
            r = np.stack(r)

            summary, _ = self.session.run([merged_summaries, _train], feed_dict={obs_ph: s,
                                                                                 act_t_ph: a,
                                                                                 return_t_ph: r,
                                                                                 learning_rate_ph: 0.005})
            self.train_writer.add_summary(summary, self.total_t)
            self.total_t += self.config.minibatch_size

        self.train = train

        def predict_values(obs):
            return self.session.run(value_out, feed_dict={obs_ph: obs})

        self.predict_values = predict_values

        self.session.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter('/tmp/train/' + str(datetime.now()) + "/", self.session.graph)

    def add_misc_summary(self, data, t):
        nice_helpers.add_misc_summary(data, t, self.train_writer)


class A2cConfig:
    def __init__(self, env_factory, session, num_steps):
        self.session = session
        self.env_factory = env_factory
        env = env_factory()

        self.num_actions = env.action_space.n

        self.frame_history_len = 1

        self.state_shape = env.observation_space.shape
        self.input_shape = tuple([self.frame_history_len] + list(self.state_shape))

        self.gamma = 0.99
        self.value_loss_constant = 1 # was 0.5
        self.regularization_constant = 0.1
        self.conv_function = atari_convnet

        self.num_actors = 16
        self.minibatch_size = 80

        self.steps_per_epoch = int(self.minibatch_size / self.num_actors)

        self.num_steps = num_steps

        self.input_data_type = np.float32
        if isinstance(env.observation_space, Box):
            self.input_data_type = np.float32

    def get_value_function(self, features):
        raise NotImplementedError

    def get_policy_function(self, features):
        raise NotImplementedError

    def get_conv_function(self, stacked_input_data):
        raise NotImplementedError

    def exploration_schedule(self):
        return utils.LinearSchedule(self.num_steps, 0.1)

    def exploration(self, t):
        return self.exploration_schedule().value(t)


class A2cConductor:
    def __init__(self, config):
        self.env_factory = config.env_factory
        self.session = config.session
        self.config = config
        self.done = False

    def run(self):
        model = A2cModel(self.config, self)

        envs = [A2cEnvironmentWrapper(i, self.env_factory, A2cAgent(self.config, model), self.config, model)
                for i in range(self.config.num_actors)]

        total_episodes_run = 0

        for i in range(int(self.config.num_steps / self.config.minibatch_size)):
            s = []
            a = []
            r = []

            for env in envs:
                env.run_batch(self.config.steps_per_epoch)
                (new_s, new_a, new_r) = env.extract_experience()
                s.extend(new_s)
                a.extend(new_a)
                r.extend(new_r)

            if i % 10 == 0 and model.episodes_log and i != 0:
                episodes = model.episodes_log
                total_episodes_run += len(episodes)
                model.episodes_log = []
                t = i * self.config.minibatch_size
                model.add_misc_summary({
                    'average_episode_reward': sum(episodes) / len(episodes),
                    'completed_episodes': total_episodes_run,
                    'epsilon': self.config.exploration(t)
                }, t)

            model.train(s, a, r)

        print("done!")
        return model

    def enjoy(self, model):
        # type: (self, A2cModel) -> None

        # make exploration very low
        self.config.exploration = lambda t: 0

        env_wrapper = A2cEnvironmentWrapper(0, self.env_factory,  A2cAgent(self.config, model), self.config, model,
                                            render=True)
        while not env_wrapper.done:
            env_wrapper.step()
