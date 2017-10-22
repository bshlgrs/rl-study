from collections import namedtuple
from datetime import datetime

import numpy as np
import tensorflow as tf
from gym.spaces import Box

import nice_helpers
import utils
from EnvironmentWrapper import EnvironmentWrapper
from ReusableAgent import ReusableAgent
from useful_neural_nets import atari_convnet


class A2cModel(object):
    def __init__(self, config, conductor):
        self.config = config
        self.conductor = conductor
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

            value_loss = nice_helpers.mean_square_error(tf.squeeze(value_out), return_t_ph)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('value_prediction_and_true_value_covariance', utils.covariance(value_out, return_t_ph))

            policy_loss = tf.reduce_mean(neg_log_p_ac * tf.stop_gradient(empirical_advantage))
            tf.summary.scalar('policy_loss', policy_loss)

            total_loss = tf.reduce_sum(policy_loss
                                       + config.value_loss_constant * value_loss
                                       - config.regularization_constant * entropy, name="total_loss")
            tf.summary.scalar('total_loss', total_loss)

        merged_summaries = tf.summary.merge_all()

        params = utils.find_trainable_variables('model')
        clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, params), 0.5)
        grads = list(zip(clipped_grads, params))

        optimizer = tf.train.RMSPropOptimizer(learning_rate_ph)
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
                                                                                 learning_rate_ph: 0.05})
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
        self.value_loss_constant = 1
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

        envs = [EnvironmentWrapper(i, self.env_factory, ReusableAgent(self.config, model), self.config, model)
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

        return model

    def enjoy(self, model):
        # type: (self, A2cModel) -> None

        # make exploration very low
        self.config.exploration = lambda t: 0

        env_wrapper = EnvironmentWrapper(0, self.env_factory, ReusableAgent(self.config, model), self.config, model,
                                         render=True)
        while not env_wrapper.done:
            env_wrapper.step()
