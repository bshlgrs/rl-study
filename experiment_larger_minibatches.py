from run_dqn_atari import *


def main():
    env = get_env(3)
    session = get_session()
    agent = DQNAgent(env, session, batch_size=256)

    agent.learn(1000001)
    env.close()

if __name__ == '__main__':
    main()