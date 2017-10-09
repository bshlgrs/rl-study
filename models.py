import datetime
import tensorflow as tf
from dqn_utils import *
from collections import namedtuple
import tensorflow.contrib.layers as layers

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])
BestAction = namedtuple('BestAction', ['action_idx', 'q_value'])


def variable_summaries(var, var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope(var_name+"-summary"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def scalar_summary(var_name):
    var = tf.get_variable(var_name, initializer=0.)
    tf.summary.scalar(var_name, var)
    return var


def scalar_summaries(names):
    return {name: scalar_summary(name) for name in names}


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
    def __init__(self, session, env, q_func=atari_model, double=True, batch_size=512):
        self.session = session

        self.q_func = q_func
        self.env = env
        self.double = double
        self.gamma = 0.99
        self.grad_norm_clipping = 10
        self.num_actions = env.action_space.n

        self.batch_size = batch_size
        self.frame_history_len = 4
        self.save_frequency = 250000

        self.log_rate = 1000

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

        self.summarized_scalars = scalar_summaries(['epsilon', 'mean_episode_reward', 'num_episodes', 'learning_rate'])

        self.train_fn = train_fn
        self.update_target_fn = update_target_fn
        self.action_choices = action_choices
        self.best_action_values = best_action_values
        self.merged_summaries = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter('/tmp/train', self.session.graph)

    def choose_best_action(self, obs):
        action_choices_fn = self.action_choices
        best_action_values_fn = self.best_action_values

        [action_choices, best_action_values] = \
            self.session.run([action_choices_fn, best_action_values_fn], feed_dict={self.obs_t_ph: np.array([obs])})
        return BestAction(action_idx=action_choices[0], q_value=best_action_values[0])

    def train(self, samples, info, t):
        obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = samples

        train_fn = self.train_fn

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
            self.learning_rate_ph: self.current_learning_rate(t)
        })

        if t % self.save_frequency == 0:
            self.save(t)

        if t % self.log_rate == 0:
            for key, value in info.items():
                self.session.run(self.summarized_scalars[key].assign(float(value)))

            merged_summaries = self.merged_summaries
            summary = self.session.run(merged_summaries)
            self.train_writer.add_summary(summary, t)

    def update_target_network(self):
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
