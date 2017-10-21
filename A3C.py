import threading
import time
from tensorflow.contrib import layers

import dqn_models
import useful_neural_nets
from utils import *


THREAD_DELAY = 0.001


def assert_shape(arr, shape):
    assert len(arr.shape) == len(shape), "Shapes should be equal: %s, %s"%(str(arr.shape), str(shape))
    for (arr_el, shape_el) in zip(arr.shape, shape):
        if shape_el is not None:
            if arr_el != shape_el:
                raise RuntimeError("Oh dear! Your array has shape %s, supposed to be %s"%(str(arr.shape), str(shape)))

    return arr


class A3cEnvironmentWrapper(threading.Thread):
    def __init__(self, env_factory, agent, config, model):
        super().__init__()
        self.env = env_factory()
        self.stop_signal = False
        self.agent = agent
        self.config = config
        self.queue = collections.deque(maxlen=config.frame_history_len + 1)
        self.model = model

        zero_state = np.zeros(self.config.state_shape)
        for i in range(config.frame_history_len + 1):
            self.queue.append(zero_state)

    def run(self):
        while not self.stop_signal:
            if len(self.model.train_queue) < self.config.max_queue_length:
                self.run_episode()
            time.sleep(THREAD_DELAY)

    def current_obs(self):
        return np.stack(list(self.queue)[1:])

    def prev_obs(self):
        return np.stack(list(self.queue)[:-1])

    def run_episode(self):
        print('running episode')
        self.queue.append(self.env.reset())
        total_reward = 0

        while True:
            time.sleep(THREAD_DELAY)
            a = self.agent.act(self.current_obs())
            s_, r, done, _ = self.env.step(a)

            total_reward += r

            if done:
                self.queue.append(np.zeros(self.config.state_shape))
            else:
                self.queue.append(s_)

            self.agent.train(self.prev_obs(), a, r, self.current_obs(), done)

            if done or self.stop_signal:
                break

        print('total reward', total_reward)
        self.agent.report_experience()


class A3cAgent:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.memory = []
        self.gamma = 0.

    def act(self, obs):
        if random.random() < self.config.exploration_schedule.value(self.model.total_t):
            return random.randrange(self.config.num_actions)
        else:
            return np.random.choice(self.config.num_actions, p=self.model.get_policy(obs))

    def train(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def report_experience(self):
        # compute n_step rewards, then send them to the master
        total_r = 0

        states = []
        actions = []
        total_rewards = []
        next_states = []
        done_mask = []

        for (s, a, r, s_, done) in reversed(self.memory):
            total_r = total_r * self.config.gamma + r

            states.append(s)
            actions.append(a)
            total_rewards.append(total_r)
            next_states.append(s_)
            done_mask.append(done)

        self.model.train_push(states, actions, total_rewards, next_states, done_mask)
        self.memory = []


class A3cModel(object):
    def __init__(self, config, conductor):
        self.config = config
        self.conductor  = conductor

        self.conv_function = self.config.conv_function
        self.session = self.config.session

        self.queue_lock = threading.BoundedSemaphore()

        self.total_t = 0

        obs_t_ph = tf.placeholder(tf.uint8, [None] + list(config.input_shape), name='obs_t_ph')
        act_t_ph = tf.placeholder(tf.int32, [None], name='act_t_ph')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='rew_t_ph')
        self.model_initialized = False

        learning_rate_ph = tf.placeholder(tf.float32, (), name="learning_rate_ph")

        num_actions = config.num_actions
        conv_function = self.conv_function

        # casting to float on GPU ensures lower data transfer times. TODO: understand this better
        obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0

        conv_layer = conv_function(obs_t_float, 'a3c_conv_func')

        with tf.variable_scope('policy'):
            policy_fc1 = layers.fully_connected(conv_layer, num_outputs=256, activation_fn=tf.nn.relu)
            policy_fc2 = layers.fully_connected(policy_fc1, num_outputs=num_actions, activation_fn=None)
            policy_out = tf.nn.softmax(policy_fc2)

        with tf.variable_scope('value'):
            value_fc1 = layers.fully_connected(conv_layer, num_outputs=256, activation_fn=tf.nn.relu)
            value_out = tf.reduce_sum(layers.fully_connected(value_fc1, num_outputs=1, activation_fn=None), axis=1)

        log_policy = tf.log(tf.reduce_sum(
            tf.one_hot(act_t_ph, num_actions) * policy_out,
            axis=1,
            keep_dims=True) + 1e-10)

        advantage = rew_t_ph - value_out

        loss_policy = -log_policy * tf.stop_gradient(advantage)
        loss_value = config.value_loss_constant * tf.square(advantage)

        entropy = config.regularization_constant * tf.reduce_sum(tf.log(policy_out + 1e-10) * policy_out, axis=1,
                                                                 keep_dims=True)
        loss_total = tf.reduce_sum(loss_policy + loss_value - entropy)
        optimizer = tf.train.RMSPropOptimizer(learning_rate_ph, decay=.99)
        minimize_step = optimizer.minimize(loss_total)

        self.train_queue = {
            's': [],
            'a': [],
            'r': [],
            's_': [],
            'done': []
        }

        self.tensors_for_get_policy = (obs_t_ph, policy_out)
        self.tensors_for_train = (obs_t_ph, act_t_ph, rew_t_ph, learning_rate_ph, minimize_step)
        self.tensors_for_get_value = (obs_t_ph, value_out)
        self.session.run(tf.global_variables_initializer())

    def get_policy(self, observation):
        (obs_t_ph, policy_out) = self.tensors_for_get_policy
        return self.session.run(policy_out, feed_dict={obs_t_ph: np.array([observation])})[0]

    def train_push(self, states, actions, n_step_rewards, next_states, done_mask):
        with self.queue_lock:
            self.train_queue['s'].extend(states)
            self.train_queue['a'].extend(actions)
            self.train_queue['r'].extend(n_step_rewards)
            self.train_queue['s_'].extend(next_states)
            self.train_queue['done'].extend(done_mask)

            self.total_t += len(states)

            print('train queue now has length %d'%(len(self.train_queue['s'])))

    def optimize(self):
        if len(self.train_queue['s']) < self.config.min_train_batch_size:
            return

        print('optimizing! t=%d'%self.total_t)

        with self.queue_lock:
            s = self.train_queue['s']
            a = self.train_queue['a']
            r = self.train_queue['r']
            s_ = self.train_queue['s_']
            done = self.train_queue['done']

            self.train_queue = {
                's': [],
                'a': [],
                'r': [],
                's_': [],
                'done': []
            }

        s = np.stack(s)
        a = np.stack(a)
        r = np.stack(r)
        s_ = np.stack(s_)
        done = np.stack(done)

        value_of_next_states = assert_shape(self.predict_values(s_), [None])
        r_including_value = assert_shape(r, [None]) + \
                            assert_shape(self.config.gamma * value_of_next_states * (1 - done), [None])

        (obs_t_ph, act_t_ph, rew_t_ph, learning_rate_ph, minimize_step) = self.tensors_for_train

        self.session.run(minimize_step, feed_dict={obs_t_ph: s, act_t_ph: a, rew_t_ph: r_including_value, learning_rate_ph: 0.005})

        if self.total_t > self.config.num_steps:
            self.conductor.done = True

    def predict_values(self, s_):
        (obs_t_ph, value_out) = self.tensors_for_get_value
        return self.session.run(value_out, feed_dict={obs_t_ph: s_})


class A3cConfig:
    def __init__(self, env, session):
        self.exploration_schedule = LinearSchedule(100000, 0.1)
        self.session = session
        self.env = env
        self.num_actions = env.action_space.n

        self.frame_history_len = 13

        self.state_shape = env.observation_space.shape
        self.input_shape = tuple([self.frame_history_len] + list(self.state_shape))

        self.min_train_batch_size = 32
        self.gamma = 0.99
        self.value_loss_constant = 0.5
        self.regularization_constant = 0.01
        self.conv_function = useful_neural_nets.atari_convnet

        self.num_steps = 75000

        self.max_queue_length = 1000


class A3cConductor:
    def __init__(self, env_factory, session):
        self.env_factory = env_factory
        self.session = session
        self.num_actors = 20
        self.num_optimizers = 2
        self.config = A3cConfig(env_factory(), session)
        self.done = False

    def run(self):
        model = A3cModel(self.config, self)

        envs = [A3cEnvironmentWrapper(self.env_factory, A3cAgent(self.config, model), self.config, model)
                for _ in range(self.num_actors)]
        optimizers = [A3cOptimizer(model) for _ in range(self.num_optimizers)]

        for env in envs:
            env.start()
        for optimizer in optimizers:
            optimizer.start()

        while not self.done:
            time.sleep(4)

        for env in envs:
            env.stop_signal = True
        for optimizer in optimizers:
            optimizer.stop_signal = True

        for env in envs:
            env.join()
        for optimizer in optimizers:
            optimizer.join()

        return model


class A3cOptimizer(threading.Thread):
    def __init__(self, model):
        super().__init__()
        self.stop_signal = False
        self.model = model

    def run(self):
        while not self.stop_signal:
            time.sleep(0)
            print('optimizer!')
            self.model.train()

