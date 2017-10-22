import A2C
import tensorflow as tf
from datetime import datetime
import nice_helpers
import numpy as np
import utils
from EnvironmentWrapper import EnvironmentWrapper
from ReusableAgent import ReusableAgent


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


class PpoConductor(A2C.A2cConductor):
    def run(self):
        model = PpoModel(self.config, self)

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

            # training time!
            model.copy_policy_to_old_policy()
            for training_step in range(5):
                model.new_policy_train_step(s, a, r)

        return model


class PpoModel(object):
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

        policy, value = self.make_functions('model', obs_float)
        old_policy, old_value = self.make_functions('old_model', obs_float)

        # Maybe this should use the cross_entropy_with_logits method
        entropy = tf.reduce_mean(-tf.log(policy + 1e-10) * policy) * num_actions

        with tf.variable_scope('losses'):
            empirical_advantage = return_t_ph - tf.stop_gradient(value)
            utils.variable_summaries(empirical_advantage, 'empirical_advantage')

            value_loss = nice_helpers.mean_square_error(tf.squeeze(value), return_t_ph)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('value_prediction_and_true_value_covariance', utils.covariance(value, return_t_ph))

            one_hot_actions = tf.one_hot(act_t_ph, num_actions)
            policy_objective = l_clip(policy, old_policy, empirical_advantage, one_hot_actions, 0.2)

            total_obj = tf.reduce_sum(policy_objective
                                       - config.value_loss_constant * value_loss -
                                       + config.regularization_constant * entropy, name="total_obj")
            tf.summary.scalar('total_obj', total_obj)

        merged_summaries = tf.summary.merge_all()

        def get_policy(observation):
            return self.session.run(policy, feed_dict={obs_ph: np.array([observation])})[0]

        self.get_policy = get_policy

        def new_policy_train_step(s, a, r):
            params = utils.find_trainable_variables('model')
            clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(-total_obj, params), 0.5)
            grads = list(zip(clipped_grads, params))

            optimizer = tf.train.RMSPropOptimizer(learning_rate_ph)
            _train = optimizer.apply_gradients(grads)

            s = np.stack(s)
            a = np.stack(a)
            r = np.stack(r)

            summary, _ = self.session.run([merged_summaries, _train], feed_dict={obs_ph: s,
                                                                                 act_t_ph: a,
                                                                                 return_t_ph: r,
                                                                                 learning_rate_ph: 0.05})
            self.train_writer.add_summary(summary, self.total_t)

        self.new_policy_train_step = new_policy_train_step

        self.copy_policy_to_old_policy = nice_helpers.hard_update_target_fn('model', 'old_model')

        def predict_values(obs):
            return self.session.run(value, feed_dict={obs_ph: obs})

        self.predict_values = predict_values

        self.session.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter('/tmp/train/' + str(datetime.now()) + "/", self.session.graph)

    def add_misc_summary(self, data, t):
        nice_helpers.add_misc_summary(data, t, self.train_writer)

    def make_functions(self, scope_name, obs_float):
        config = self.config

        with tf.variable_scope(scope_name):
            conv_out = config.get_conv_function(obs_float)

            with tf.variable_scope('policy'):
                policy_out = config.get_policy_function(conv_out)

            with tf.variable_scope('value'):
                value_out = config.get_value_function(conv_out)

        return [policy_out, value_out]