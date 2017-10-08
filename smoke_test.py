from run_dqn_atari import *


def main():
    env = get_env(3)
    session = get_session()
    agent = DQNAgent(env, session, batch_size=512)
    agent.learning_starts = 1
    agent.learning_freq = 1

    agent.learn(3)


    env.close()

if __name__ == '__main__':
    main()