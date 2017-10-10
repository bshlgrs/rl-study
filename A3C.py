import models
from dqn_utils import *
import gym.spaces


class A3CModel(object):
    def __init__(self, session, env, conv_function=None):
        self.session = session
        self.env = env
        self.num_actions = env.action_space.n

        self.frame_history_len = 3

        self.conv_function = conv_function or models.atari_convnet

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




    def sync_params(self):
        pass

    def get_action(self, observation):
        pass

    def evaluate_value_function_on_obs(self, obs):
        pass


class A3CAgent:
    def __init__(self, env, session, max_episode_timesteps=100000, gamma=0.99, conv_function=None):
        self.env = env
        self.session = session
        self.max_episode_timesteps = max_episode_timesteps
        self.model = A3CModel(conv_function=conv_function)
        self.gamma = 0.99


        assert type(self.env.observation_space) == gym.spaces.Box
        assert type(self.env.action_space) == gym.spaces.Discrete

    def learn(self):
        pass

    def child_thread_loop_function(self):
        self.model.sync_params()

        replay_buffer = ReplayBuffer(self.max_episode_timesteps, self.model.frame_history_len)

        last_obs = self.env.reset()

        done = False
        i = 0

        for i in range(self.max_episode_timesteps):
            idx = replay_buffer.store_frame(last_obs)

            action = self.model.get_action(replay_buffer.encode_recent_observation())

            obs, reward, done, info = self.env.step(action)

            replay_buffer.store_effect(idx, action, reward, done)
            last_obs = obs

            if done:
                break

        self.increase_global_counter(i)

        if done:
            r = 0
        else:
            r = self.model.evaluate_value_function_on_obs(replay_buffer.encode_recent_observation())

        delta_theta = None
        delta_theta_v = None
        for i in range(i, 0, -1):
            r = self.gamma * r + replay_buffer.reward[i]

            # get the gradients for theta
            # get the gradients for theta_v

        self.update_global_parameters(delta_theta, delta_theta_v)



    def child_thread_loop(self):
        while True:
            self.child_thread_loop()

    def increase_global_counter(self, i):
        pass # TODO

    def update_global_parameters(self, delta_theta, delta_theta_v):
        pass