import models
from run_dqn_atari import *
import os


def main():
    print("your git commit is ")
    os.system("git rev-parse HEAD")

    # options are: Beam Rider, Breakout, Enduro, Pong, Qbert, Seaquest, Space Invaders
    env = get_env(3)
    session = get_session()
    agent = DQNAgent(env, session, batch_size=512, q_func=models.dueling_atari_model)

    agent.learn(2000001)
    env.close()


if __name__ == '__main__':
    main()
