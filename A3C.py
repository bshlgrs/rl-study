from dqn_utils import *


class A3CModel(object):
    def __init__(self):
        pass

    def sync_params(self):
        pass

    def get_action(self, observation):
        pass

    def evaluate_value_function_on_obs(self, obs):
        pass


class A3CAgent:
    def __init__(self, env, session, max_episode_timesteps=100000, gamma=0.99):
        self.env = env
        self.session = session
        self.max_episode_timesteps = max_episode_timesteps
        self.model = A3CModel()
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