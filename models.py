import datetime
import tensorflow as tf
from dqn_utils import *
from collections import namedtuple
import tensorflow.contrib.layers as layers

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])
BestAction = namedtuple('BestAction', ['action_idx', 'q_value'])


def scalar_summaries(names):
    for name in names:
        scalar_summary(name)


def atari_convnet(img_in, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("convnet", reuse=reuse):
            conv1 = layers.convolution2d(img_in, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            conv2 = layers.convolution2d(conv1, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            conv3 = layers.convolution2d(conv2, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
    return layers.flatten(conv3)


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = atari_convnet(img_in, scope, reuse)

        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def dueling_atari_model(img_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        conv_out = atari_convnet(img_in, scope, reuse)

        with tf.variable_scope('unnormalized_advantage'):
            advantage_fc1 = layers.fully_connected(conv_out, num_outputs=512, activation_fn=tf.nn.relu)
            unnormalized_advantage = layers.fully_connected(advantage_fc1, num_outputs=num_actions, activation_fn=None)

        with tf.variable_scope('value_function'):
            value_fc1 = layers.fully_connected(conv_out, num_outputs=512, activation_fn=tf.nn.relu)
            value = layers.fully_connected(value_fc1, num_outputs=1, activation_fn=None)

        normalized_advantage = unnormalized_advantage - tf.expand_dims(
            tf.reduce_mean(unnormalized_advantage, axis=1), axis=1)

        q_func = value + normalized_advantage

    # print('normalized_advantage shape:', normalized_advantage.get_shape())
    # print('q func shape:', q_func.get_shape())
    return q_func


def dueling_atari_model_2(img_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        conv_out = atari_convnet(img_in, scope, reuse)
        fc_1 = layers.fully_connected(conv_out, num_outputs=512, activation_fn=tf.nn.relu)

        with tf.variable_scope('unnormalized_advantage'):
            advantage_fc2 = layers.fully_connected(fc_1, num_outputs=32, activation_fn=tf.nn.relu)
            unnormalized_advantage = layers.fully_connected(advantage_fc2, num_outputs=num_actions, activation_fn=None)

        with tf.variable_scope('value_function'):
            value_fc1 = layers.fully_connected(fc_1, num_outputs=32, activation_fn=tf.nn.relu)
            value = layers.fully_connected(value_fc1, num_outputs=1, activation_fn=None)

        normalized_advantage = unnormalized_advantage - tf.expand_dims(
            tf.reduce_mean(unnormalized_advantage, axis=1), axis=1)

        q_func = value + normalized_advantage

    # print('normalized_advantage shape:', normalized_advantage.get_shape())
    # print('q func shape:', q_func.get_shape())
    return q_func


class Model:
    def __init__(self, session, env, double=True, batch_size=512, q_func=None):
        self.session = session

        self.q_func = q_func or atari_model
        self.env = env
        self.double = double
        self.gamma = 0.99
        self.grad_norm_clipping = 10
        self.num_actions = env.action_space.n

        self.batch_size = batch_size
        self.frame_history_len = 4
        self.save_frequency = 250000

        img_h, img_w, img_c = env.observation_space.shape
        self.input_shape = (img_h, img_w, self.frame_history_len * img_c)

        self.start_time = datetime.datetime.now()

        self.obs_t_ph = tf.placeholder(tf.uint8, [None] + list(self.input_shape), name='obs_t_ph')
        self.act_t_ph = tf.placeholder(tf.int32, [None], name='act_t_ph')
        self.rew_t_ph = tf.placeholder(tf.float32, [None], name='rew_t_ph')
        self.obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(self.input_shape), name='obs_tp1_ph')
        self.done_mask_ph = tf.placeholder(tf.float32, [None], name='done_mask_ph')
        self.model_initialized = False

        self.learning_rate_ph = tf.placeholder(tf.float32, (), name="learning_rate_ph")

        num_actions = self.num_actions
        q_func = self.q_func

        # casting to float on GPU ensures lower data transfer times. TODO: understand this better
        obs_t_float = tf.cast(self.obs_t_ph, tf.float32) / 255.0
        obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

        q_values_all_actions = self.q_func(obs_t_float, num_actions, 'q_func', reuse=False)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

        target_next_q_values = q_func(obs_tp1_float, num_actions, 'target_q_func', reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

        if self.double:
            all_values_of_next_states = q_func(obs_tp1_float, num_actions, 'q_func', reuse=True)
            chosen_actions = tf.one_hot(tf.argmax(all_values_of_next_states, axis=1), num_actions, name='chosen_actions_onehot')
            value_of_next_states = tf.reduce_sum(chosen_actions * target_next_q_values, axis=1,
                                                 name='value_of_next_states')
        else:
            value_of_next_states = tf.reduce_max(target_next_q_values, axis=1, name='value_of_next_states')

        y = tf.add(self.rew_t_ph, self.gamma * (1 - self.done_mask_ph) * value_of_next_states, name='y')

        q_values_for_actions_taken = tf.reduce_sum(tf.one_hot(self.act_t_ph, num_actions) * q_values_all_actions,
                                                   axis=1,
                                                   name='q_values_for_actions_taken')

        action_choices = tf.argmax(q_values_all_actions, axis=1, name='action_choices')
        best_action_values = tf.reduce_max(q_values_all_actions, axis=1, name='best_action_values')

        variable_summaries(best_action_values, 'best_action_values')

        total_error = tf.reduce_mean((y - q_values_for_actions_taken) ** 2, name='total_error')

        optimizer = self.get_optimizer_spec().constructor(
            learning_rate=self.learning_rate_ph,
            **self.get_optimizer_spec().kwargs)
        # construct optimization op (with gradient clipping)
        train_fn = minimize_and_clip(optimizer, total_error,
                                     var_list=q_func_vars, clip_val=self.grad_norm_clipping)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        update_target_fn = tf.group(*update_target_fn, name='update_target_fn')

        variable_summaries(q_values_all_actions, 'q_values_all_actions')

        scalar_summaries(['epsilon', 'mean_episode_reward', 'num_episodes', 'learning_rate'])

        merged_summaries = tf.summary.merge_all()

        self.train_fn = train_fn
        self.update_target_fn = update_target_fn
        self.action_choices = action_choices
        self.best_action_values = best_action_values
        self.merged_summaries = merged_summaries

        self.train_writer = tf.summary.FileWriter('/tmp/train', self.session.graph)

    def choose_best_action(self, obs):
        action_choices_fn = self.action_choices
        best_action_values_fn = self.best_action_values

        [action_choices, best_action_values] = \
            self.session.run([action_choices_fn, best_action_values_fn], feed_dict={self.obs_t_ph: np.array([obs])})
        return BestAction(action_idx=action_choices[0], q_value=best_action_values[0])

    def train(self, samples, t):
        obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = samples

        train_fn = self.train_fn
        merged_summaries = self.merged_summaries

        if not self.model_initialized:
            print('initializing model')
            initialize_interdependent_variables(self.session, tf.global_variables(), {
                self.obs_t_ph: obs_t_batch,
                self.obs_tp1_ph: obs_tp1_batch,
            })
            self.model_initialized = True

        _, summary = self.session.run([train_fn, merged_summaries], feed_dict={
            self.obs_t_ph: obs_t_batch,
            self.obs_tp1_ph: obs_tp1_batch,
            self.act_t_ph: act_batch,
            self.rew_t_ph: rew_batch,
            self.done_mask_ph: done_mask,
            self.learning_rate_ph: self.current_learning_rate(t)
        })

        self.train_writer.add_summary(summary, t)

        if t % self.save_frequency == 0:
            self.save(t)

    def update_target_network(self):
        print('updating target fn')
        self.session.run(self.update_target_fn)

    def current_learning_rate(self, t):
        return self.get_optimizer_spec().lr_schedule.value(t)

    @memoized
    def get_optimizer_spec(self):
        # This is just a rough estimate
        num_iterations = float(2e6)

        lr_multiplier = self.batch_size / 32.0
        lr_schedule = PiecewiseSchedule([
            (0, 1e-4 * lr_multiplier),
            (num_iterations / 10, 1e-4 * lr_multiplier),
            (num_iterations / 2, 5e-5 * lr_multiplier),
        ],
            outside_value=5e-5 * lr_multiplier)

        return OptimizerSpec(
            constructor=tf.train.AdamOptimizer,
            kwargs=dict(epsilon=1e-4),
            lr_schedule=lr_schedule
        )

    @memoized
    def get_saver(self):
        return tf.train.Saver()

    def save(self, t):
        print('saving!')
        self.get_saver().save(self.session,
                              "/home/paperspace/models/model-started-at-%s-t-%d.ckpt" % (self.start_time, t))
        print('saving done')

    def log_agent_info(self, info):
        if self.model_initialized:
            with tf.variable_scope('scalar-summary', reuse=True):
                for key, value in info.items():
                    self.session.run(tf.get_variable(key).assign(float(value)))
