import datetime
import tensorflow as tf
from dqn_utils import *
from collections import namedtuple
import tensorflow.contrib.layers as layers

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def duelling_atari_model(img_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope('convnet'):
            conv_layer_1 = layers.convolution2d(img_in, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            conv_layer_2 = layers.convolution2d(conv_layer_1, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            conv_layer_3 = layers.convolution2d(conv_layer_2, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

        conv_flattened = layers.flatten(conv_layer_3)

        with tf.variable_scope('value'):
            value_fc1 = layers.fully_connected(conv_flattened, num_outputs=512, activation_fn=tf.nn.relu)
            value_out = tf.reduce_sum(layers.fully_connected(value_fc1, num_outputs=1, activation_fn=tf.nn.relu), axis=1)

        with tf.variable_scope('advantage'):
            advantage_fc1 = layers.fully_connected(conv_flattened, num_outputs=512, activation_fn=tf.nn.relu)
            advantage_out = layers.fully_connected(advantage_fc1, num_outputs=num_actions, activation_fn=None)

        correction = tf.reduce_mean(advantage_out, axis=1)

    return advantage_out + tf.tile(tf.expand_dims(value_out - correction, 1), [1, num_actions])


class Model:
    def __init__(self, session, env, q_func=atari_model, double=True):
        self.session = session

        self.q_func = q_func
        self.env = env
        self.double = double
        self.gamma = 0.99
        self.grad_norm_clipping = 10
        self.num_actions = env.action_space.n

        self.batch_size = 128
        self.target_update_freq = 10000
        self.frame_history_len = 4
        self.save_frequency = 250000

        img_h, img_w, img_c = env.observation_space.shape
        self.input_shape = (img_h, img_w, self.frame_history_len * img_c)

        self.start_time = datetime.datetime.now()

        self.obs_t_ph = tf.placeholder(tf.uint8, [None] + list(self.input_shape))
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        self.obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(self.input_shape))
        self.done_mask_ph = tf.placeholder(tf.float32, [None])
        self.model_initialized = False

        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        self.optimizer = self.get_optimizer_spec().constructor(
            learning_rate=self.learning_rate,
            **self.get_optimizer_spec().kwargs)

    @memoized
    def build_model(self):
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
            chosen_actions = tf.one_hot(tf.argmax(all_values_of_next_states, axis=1), num_actions)
            value_of_next_states = tf.reduce_sum(chosen_actions * target_next_q_values, axis=1)
        else:
            value_of_next_states = tf.reduce_max(target_next_q_values, axis=1)

        y = self.rew_t_ph + self.gamma * (1 - self.done_mask_ph) * value_of_next_states
        q_values_for_actions_taken = tf.reduce_sum(tf.one_hot(self.act_t_ph, num_actions) * q_values_all_actions, axis=1)

        deterministic_actions = tf.argmax(q_values_all_actions, axis=1)

        total_error = tf.reduce_mean((y - q_values_for_actions_taken) ** 2)

        # construct optimization op (with gradient clipping)

        train_fn = minimize_and_clip(self.optimizer, total_error,
                                     var_list=q_func_vars, clip_val=self.grad_norm_clipping)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        update_target_fn = tf.group(*update_target_fn)

        return [train_fn, update_target_fn, deterministic_actions]

    @memoized
    def get_train_fn(self):
        return self.build_model()[0]

    @memoized
    def get_update_target_fn(self):
        return self.build_model()[1]

    @memoized
    def get_deterministic_actions(self):
        return self.build_model()[2]

    def choose_best_action(self, obs):
        return self.session.run(self.get_deterministic_actions(), feed_dict={self.obs_t_ph: np.array([obs])})[0]

    def train(self, samples, t):
        obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = samples

        train_fn = self.get_train_fn()

        if not self.model_initialized:
            print('initializing model')
            initialize_interdependent_variables(self.session, tf.global_variables(), {
                self.obs_t_ph: obs_t_batch,
                self.obs_tp1_ph: obs_tp1_batch,
            })
            self.model_initialized = True

        self.session.run(train_fn, feed_dict={
            self.obs_t_ph: obs_t_batch,
            self.obs_tp1_ph: obs_tp1_batch,
            self.act_t_ph: act_batch,
            self.rew_t_ph: rew_batch,
            self.done_mask_ph: done_mask,
            self.learning_rate: self.get_optimizer_spec().lr_schedule.value(t) * 4
        })

        if t % self.target_update_freq == 0:
            self.session.run(self.get_update_target_fn())

        if t % self.save_frequency == 0:
            print('saving!')
            self.get_saver().save(self.session,
                            "/home/paperspace/models/model-started-at-%s-t-%d.ckpt" % (self.start_time, t))
            print('saving done')

    @memoized
    def get_optimizer_spec(self):
        # This is just a rough estimate
        num_iterations = float(2e6) / 4.0

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