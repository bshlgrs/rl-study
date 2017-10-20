import random
from collections import namedtuple, deque

from tensorflow.contrib import layers
import tensorflow as tf

import utils
import numpy as np
from models import atari_convnet


THREAD_DELAY = 0.001

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
        # if random.random() < 0.01:
        #     print('Episode reward at total timestep %d is %f' % (self.model.total_t, self.episode_reward))
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
        if random.random() < self.config.exploration_schedule.value(self.model.total_t):
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

        if config.input_data_type == np.float32:
            obs_ph = tf.placeholder(tf.float32, [None] + list(config.input_shape), name='obs')
            obs_tp1 = tf.placeholder(tf.float32, [None] + list(config.input_shape), name='obs_tp1')
            obs_float = obs_ph
            obs_tp1_float = obs_tp1
        else:
            obs_ph = tf.placeholder(tf.uint8, [None] + list(config.input_shape), name='obs')
            obs_tp1 = tf.placeholder(tf.uint8, [None] + list(config.input_shape), name='obs_tp1')
            obs_float = tf.cast(obs_ph, tf.float32) / 255.0
            obs_tp1_float = tf.cast(obs_tp1, tf.float32) / 255.0

        act_t_ph = tf.placeholder(tf.int32, [None], name='act_t_ph')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='rew_t_ph')
        self.model_initialized = False

        action_probabilities = tf.reduce_mean(tf.one_hot(act_t_ph, config.num_actions), axis=0)

        for i in range(config.num_actions):
            tf.summary.scalar('action_%d_prob'%i, action_probabilities[i])

        learning_rate_ph = tf.placeholder(tf.float32, (), name="learning_rate_ph")

        num_actions = config.num_actions
        conv_function = self.conv_function

        with tf.variable_scope('model'):
            conv_layer = conv_function(obs_float, 'a3c_conv_func')

            with tf.variable_scope('policy'):
                # policy_fc1 = layers.fully_connected(conv_layer, num_outputs=16, activation_fn=tf.nn.relu)
                policy_fc2 = layers.fully_connected(conv_layer, num_outputs=num_actions, activation_fn=None)
                policy_out = tf.nn.softmax(policy_fc2)

                utils.variable_summaries(policy_fc2, 'policy_fc2')

            with tf.variable_scope('value'):
                value_fc1 = layers.fully_connected(conv_layer, num_outputs=16, activation_fn=tf.nn.relu)
                value_out = tf.reduce_sum(layers.fully_connected(value_fc1, num_outputs=1, activation_fn=None), axis=1)

                utils.variable_summaries(value_out, 'value')

        neg_log_p_ac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_out, labels=act_t_ph)
        empirical_advantage = rew_t_ph - value_out
        utils.variable_summaries(empirical_advantage, 'empirical_advantage')

        policy_loss = tf.reduce_mean(neg_log_p_ac * tf.stop_gradient(empirical_advantage))
        tf.summary.scalar('policy_loss', policy_loss)
        value_loss = tf.reduce_mean(tf.square(tf.squeeze(value_out) - rew_t_ph)) # previously this was empirical_advantage
        tf.summary.scalar('value_loss', value_loss)
        tf.summary.scalar('value_bias', tf.reduce_mean(tf.squeeze(value_out) - rew_t_ph))
        tf.summary.scalar('value_prediction_and_true_value_covariance', utils.covariance(value_out, rew_t_ph))

        entropy = tf.reduce_mean(-tf.log(policy_out + 1e-10) * policy_out) * num_actions
        tf.summary.scalar('entropy', entropy)
        total_loss = tf.reduce_sum(policy_loss +
                                   config.value_loss_constant * value_loss -
                                   config.regularization_constant * entropy, name="total_loss")
        tf.summary.scalar('total_loss', total_loss)

        self.merged_summaries = tf.summary.merge_all()

        params = utils.find_trainable_variables('model')
        clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, params), 0.5)
        grads = list(zip(clipped_grads, params))

        optimizer = tf.train.RMSPropOptimizer(learning_rate_ph, decay=.99, epsilon=1e-5)
        _train = optimizer.apply_gradients(grads)

        self.tensors_for_get_policy = (obs_ph, policy_out)
        self.tensors_for_train = (obs_ph, act_t_ph, rew_t_ph, learning_rate_ph, _train, neg_log_p_ac)
        self.tensors_for_get_value = (obs_ph, value_out)
        self.session.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter('/tmp/train', self.session.graph)

    def get_policy(self, observation):
        (obs_ph, policy_out) = self.tensors_for_get_policy
        policy = self.session.run(policy_out, feed_dict={obs_ph: np.array([observation])})[0]

        return policy

    def train(self, s, a, r):
        s = np.stack(s)
        a = np.stack(a)
        r = np.stack(r)

        (obs_ph, act_t_ph, rew_t_ph, learning_rate_ph, _train, neg_log_p_ac) = self.tensors_for_train
        merged_summaries = self.merged_summaries

        summary, _ = self.session.run([merged_summaries, _train], feed_dict={obs_ph: s,
                                                                             act_t_ph: a,
                                                                             rew_t_ph: r,
                                                                             learning_rate_ph: 0.05})
        self.train_writer.add_summary(summary, self.total_t)
        self.total_t += self.config.minibatch_size

    def predict_values(self, obs):
        (obs_t_ph, value_out) = self.tensors_for_get_value
        return self.session.run(value_out, feed_dict={obs_t_ph: obs})


class A2cConfig:
    def __init__(self, env, session):
        self.exploration_schedule = utils.LinearSchedule(2500000, 0.1)
        self.session = session
        self.env = env
        self.num_actions = env.action_space.n

        self.frame_history_len = 1

        self.state_shape = env.observation_space.shape
        self.input_shape = tuple([self.frame_history_len] + list(self.state_shape))

        self.gamma = 0.99
        self.value_loss_constant = 0.5
        self.regularization_constant = 0.01
        self.conv_function = atari_convnet

        self.num_actors = 16
        self.minibatch_size = 80

        self.steps_per_epoch = int(self.minibatch_size / self.num_actors)

        self.num_steps = 2500000

        self.input_data_type = None


class A2cConductor:
    def __init__(self, env_factory, session):
        self.env_factory = env_factory
        self.session = session

        self.config = A2cConfig(env_factory(), session)
        self.done = False

    def run(self):
        model = A2cModel(self.config, self)

        envs = [A2cEnvironmentWrapper(i, self.env_factory, A2cAgent(self.config, model), self.config, model)
                for i in range(self.config.num_actors)]

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

            if i % 10 == 0 and model.episodes_log:
                episodes = model.episodes_log
                model.episodes_log = []

                print(i * self.config.minibatch_size, sum(episodes) / len(episodes))
            model.train(s, a, r)

        print("done!")
        return model

    def enjoy(self, model):
        # type: (self, A2cModel) -> None
        self.config.exploration_schedule = utils.LinearSchedule(1, 0)
        env_wrapper = A2cEnvironmentWrapper(0, self.env_factory,  A2cAgent(self.config, model), self.config, model,
                                            render=True)
        while not env_wrapper.done:
            env_wrapper.step()
