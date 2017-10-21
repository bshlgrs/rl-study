import dqn_models
from run_dqn_atari import *


def main():
    env = get_env(3)
    session = get_session()
    agent = DQNAgent(env, session, batch_size=7, q_func=dqn_models.dueling_atari_model)
    agent.learning_starts = 1
    agent.learning_freq = 1
    agent.log_rate = 2
    agent.learn(10)

    env.close()

if __name__ == '__main__':
    main()