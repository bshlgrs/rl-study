import random

from gym import wrappers
import os.path as osp

from utils import *
from atari_wrappers import *
from dqn import DQNAgent


def atari_learn(num_timesteps, task_idx=3):
    env = get_env(task_idx)
    session = get_session()
    agent = DQNAgent(env, session)
    agent.learn(num_timesteps)
    env.close()


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def get_env(task_idx, monitor=True):
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[task_idx]

    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(0)
    env.seed(0)

    expt_dir = '/tmp/hw3_vid_dir2/'
    if monitor:
        env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env


def main():
    atari_learn(num_timesteps=int(2e6))


if __name__ == "__main__":
    main()
