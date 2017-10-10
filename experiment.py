import models
from run_dqn_atari import *
import os


def main():
    print("your git commit is ")
    os.system("git rev-parse HEAD")

    env = get_env(1)
    session = get_session()
    agent = DQNAgent(env, session, batch_size=512, q_func=models.duelling_atari_model)

    agent.learn(5000001)
    env.close()


if __name__ == '__main__':
    main()