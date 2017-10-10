from run_dqn_atari import *
import os


def main():
    print("your git commit is ")
    os.system("git rev-parse HEAD")

    env = get_env(1)
    session = get_session()
    agent = DQNAgent(env, session, batch_size=512)

    agent.learn(5000001)
    env.close()


if __name__ == '__main__':
    main()